#this file is the data structure of ML data (x,y)

from __future__ import division
from itertools import count

from SUGAR_PF import main as runPF
from SUGAR_PF import clearPF
from process_raw import getBusNameBook
from process_raw import readRawFile
import networkx as nx
from global_vars import global_vars

import sys
from Buses import Buses
from classes.Generators import Generators
from classes.Slack import Slack
from classes.class_gen_limits import GenLimits
from itertools import count
import numpy as np
import os

import random
import numpy as np
import math
import copy

#import pandapower
#import pandapower.estimation

import csv


    
def calcI(Vr,Vi,G,B):
    Ir = G*Vr-B*Vi
    Ii = B*Vr+G*Vi
    return Ir, Ii

def calcPQ(Vr, Vi, Ir, Ii):
    P = Vr*Ir + Vi*Ii
    Q = Vi*Ir - Vr*Ii
    return P, Q

def update_bus_injection(bus,generator,slack,load,shunt,shunt_included = False):
    """may need infeasibility source to calculate injections?"""
    #init
    for ele in bus:
        ele.Pinj = 0.0
        ele.Qinj = 0.0
        ele.Qshunt = 0.0
    #add shunt P,Q: 
    for ele in shunt:
        Bus = ele.Bus
        bus_key = Buses.bus_key_[Bus]
        Vr = bus[bus_key].Vr_init
        Vi = bus[bus_key].Vi_init
        Ir, Ii = calcI(Vr,Vi,ele.G_pu, ele.B_pu)
        Pshunt, Qshunt = calcPQ(Vr, Vi, Ir, Ii)
        if shunt_included:
            bus[bus_key].Pinj = bus[bus_key].Pinj + Pshunt
            bus[bus_key].Qinj = bus[bus_key].Qinj + Qshunt
        bus[bus_key].Qshunt = Qshunt
    #add gen PQ
    for ele in generator:
        Bus = ele.Bus
        bus_key = Buses.bus_key_[Bus]
        Pgen = ele.P
        Qgen = ele.Qinit
        bus[bus_key].Pinj = bus[bus_key].Pinj - Pgen # use '-' since the direction is from node to ground, and the generator PQ direction is outflow
        bus[bus_key].Qinj = bus[bus_key].Qinj - Qgen      
    #add slack PQ:
    for ele in slack:
        Bus = ele.Bus
        bus_key = Buses.bus_key_[Bus]
        Psl = ele.Pinit
        Qsl = ele.Qinit
        bus[bus_key].Pinj = bus[bus_key].Pinj - Psl # use '-' since the direction is inject, and the generator PQ direction is outflow 
        bus[bus_key].Qinj = bus[bus_key].Qinj - Qsl         
    #add load PQ
    for ele in load:
        Bus = ele.Bus
        bus_key = Buses.bus_key_[Bus]
        Pload = ele.P
        Qload = ele.Q
        bus[bus_key].Pinj = bus[bus_key].Pinj + Pload # use '+' since the both direction is inject
        bus[bus_key].Qinj = bus[bus_key].Qinj + Qload   
    """calculate Ir, Ii, and convert injection from p.u to MVA"""
    for ele in bus:
        Bus = ele.Bus
        if isinstance(ele.Pinj,np.ndarray):
            ele.Pinj = ele.Pinj.reshape(-1)[0]
        if isinstance(ele.Qinj,np.ndarray):
            ele.Qinj = ele.Qinj.reshape(-1)[0]
        if isinstance(ele.Vr_init,np.ndarray):
            ele.Vr_init = ele.Vr_init.reshape(-1)[0]
        if isinstance(ele.Vi_init,np.ndarray):
            ele.Vi_init = ele.Vi_init.reshape(-1)[0]
        if isinstance(ele.Vm_init,np.ndarray):
            ele.Vm_init = ele.Vm_init.reshape(-1)[0]
        ele.Irinj = (ele.Pinj*ele.Vr_init + ele.Qinj*ele.Vi_init) / (ele.Vm_init)**2
        ele.Iiinj = (ele.Pinj*ele.Vi_init - ele.Qinj*ele.Vr_init) / (ele.Vm_init)**2            
        #ele.Pinj_MW = ele.Pinj*global_vars.MVAbase
        #ele.Qinj_MVAR = ele.Qinj*global_vars.MVAbase
        #print(Bus,ele.Vm_init, ele.Pinj, ele.Qinj)    
    
    return bus

class simulated_data():
    def __init__(self, case_name, c_type, argvs, manipulate = None, contingency = None, auto_create=True):
        np.random.seed()
        #raw case data file
        self.case_name = case_name # a string
        self.Niter = dict() #iterations needed to solve PF, e.g. self.Niter['methodname'] = 10
        self.x = dict() #store the features of node / line / xfmr / etc
        
        self.name2bus_book, self.bus2name_book = getBusNameBook('./raw/'+self.case_name)
        
        #define pre-contingency case 
        if not manipulate:
            """self.manipulate defines precontingency system, compared with the raw case data"""
            self.manipulate = dict()  
            self.manipulate['open branch'] = [] #open non transformer branches
            self.manipulate['close branch'] = [] #not used
            self.manipulate['open xfmr'] = [] #open transformer branches
            self.manipulate['close xfmr'] = [] #
            self.manipulate['bus off'] = [] #substation outage, bus is offline
            self.manipulate['gen on'] = [] #turn on an inactive gen, not used
            self.manipulate['gen off'] = [] #disconnect gen, gen outage
            self.manipulate['load on'] = [] #turn on an inactive load, not used
            self.manipulate['load off'] = [] #disconnect load, load outage
            self.manipulate['lf'] = 1 #load factor
            self.manipulate['gf'] = 1 #gen factor
            self.manipulate['load scaling'] = [] #[[bus1, factor1],[bus2, factor2]]
            self.manipulate['gen scaling'] = [] #
            self.create_lf()
            self.manipulate['gf'] = self.manipulate['lf']
            self.create_topology()
            clearPF()
        else:
            self.manipulate = manipulate
        if contingency:
            self.contingency = contingency

        self.v = dict()

        if auto_create:
            #pre contingency simulation
            self.feasible = False #the last simulation should be a feasible case
            self.simulate(argvs,'pre')

            while not self.feasible:
                self.create_lf()
                self.create_topology()
                clearPF()
                self.simulate(argvs,'pre')
                clearPF()
            self.feasible = False #post contingency should be a feasible case

            #define contingency
            if not contingency:
                # 'genoutage' # substationoutage, genoutage, lineoutage
                print(c_type)
                self.create_contingency(c_type)
                #self.contingency = {'type':c_type, 'location': [], 'parameter':[]}
            clearPF()
            #post contingency simulation
            self.simulate(argvs,'post')
            clearPF()

            if not self.feasible:
                enable_homotopy = True
                self.simulate(argvs, 'post','file',enable_homotopy)
                clearPF()


        
    def create_lf(self, mode='random'):
        """sample a random scalar to scale loads"""
        if mode == 'random':
            self.manipulate['lf'] = np.random.uniform(low=0.95, high=1.05, size=(1))[0]
        
    def create_gf(self,mode='random'):
        """sample a random scalar to scale generations"""
        if mode == 'random':
            self.manipulate['lf'] = np.random.uniform(low=0.95, high=1.05, size=(1))[0]
            
    def create_topology(self,br2open=[], br2close=[]):
        """randomly sample branches(lines) that will be openned/closed from the raw case data"""
        parsed_data, _, _, _ = readRawFile('./raw/'+self.case_name, 1, None)
        branch = parsed_data['branches']
        K_topo = np.random.randint(0,2) #disconnect how many lines from the raw data
        br2open = []
        if K_topo >0:
            br2open = [[branch[i].from_bus,branch[i].to_bus] for i in np.random.choice(len(branch), K_topo, replace=False)]
            print("samepled %d lines to open and create a new pre-contingency topology"%K_topo,br2open)

        self.manipulate['open branch'] = br2open
        
        br2close = []
        self.manipulate['close branch'] = br2close
        
        
    def create_contingency(self, c_type):
        #self.contingency = {'type':'lineoutage','location':[[3,4]], 'parameter':[]} #location stores the outage branch id in raw file
        if c_type == 'lineoutage':
            #randomly select k lines of pre-contingency case to disconnect
            k = 5 #N-k contingency of the type
            if_islanded = True
            while if_islanded:
                outage_br = [[self.x['line feature'][i][0],self.x['line feature'][i][1]] for i in np.random.choice(len(self.x['line feature']), k, replace=False)]
                """check if network is islanded"""
                branch_edge = [(ele.from_bus,ele.to_bus)
                               for ele in self.branch_pre if [ele.from_bus,ele.to_bus] not in outage_br]
                xfmr_edge = [(ele.from_bus,ele.to_bus)
                             for ele in self.xfmr_pre]
                pairs = branch_edge + xfmr_edge
                graph = nx.from_edgelist(pairs)
                islands = list(nx.connected_components(graph))
                if_islanded = len(islands)!=1
                if if_islanded:
                    print('network islanded, recreate contingnency')
                else:
                    print('network not islanded, contingency created')
            self.contingency = {'type':c_type, 'location': outage_br, 'parameter':[]}
        elif c_type == 'MadIoT':
            nload = len(self.load_pre)
            ngen = len(self.gen_pre)
            K = round(0.5*nload) #attack K loads
            Ploads = [ele.P for ele in self.load_pre]
            ids2attack = np.argsort(Ploads)[::-1]#ids that rands Pload in descending order
            ids2attack = ids2attack[0:K] #choose the top K largest loads
            load2attack = [self.load_pre[id].Bus for id in ids2attack] #bus names of loads to attack
            # for ele in self.load_pre:
            #     print('madiot load',ele.Bus,ele.P)
            lf2attack = 1.2 #1.2: increase the attacked loads by 20%
            Pgen_total = sum([ele.P for ele in self.gen_pre]) + sum([ele.Pinit for ele in self.slack_pre])
            dPload = sum([ele.P * (lf2attack - 1) for ele in self.load_pre if ele.Bus in load2attack])
            gf2droop = 1 + dPload / Pgen_total
            print('--madiot: among %d loads, %d gens' % (nload, ngen),' attack %d loads by %f, accumu gf=%f'%(K,lf2attack,gf2droop))
            self.contingency = {'type':c_type, 'location': load2attack, 'parameter': [lf2attack,gf2droop]}
            #print('create contingency: MadIoT at load bus #', load2attack, ' by ', para)
        elif c_type == 'genoutage':
            ngen = len(self.gen_pre)
            K = 5  # attack K gens, N-5, N-10
            """choose the top K largest gen?"""
            #Pgens = [ele.P for ele in self.gen_pre]
            #ids2attack = np.argsort(Pgens)[::-1]  # ids that rands Pload in descending order
            #ids2attack = ids2attack[0:K]  # choose the top K largest loads
            #gen2attack = [self.gen_pre[id].Bus for id in ids2attack]  # bus names of gen to attack
            """randomly pick K generators"""
            gen_buses = [ele.Bus for ele in self.gen_pre] #bus names of all gen buses, except slacks
            #gen_buses.extend([ele.Bus for ele in self.slack_pre]) #add slack bus
            gen2attack = np.random.choice(gen_buses,K, replace=False)
            #print('totally %d gens, pick %d. ' % (ngen + len(self.slack_pre), K), gen_buses, gen2attack)
            Pgen_remain = sum([ele.P for ele in self.gen_pre if ele.Bus not in gen2attack]) + \
                          sum([ele.Pinit for ele in self.slack_pre if ele.Bus not in gen2attack])
            dPgen = sum([ele.P for ele in self.gen_pre if ele.Bus in gen2attack]) + \
                    sum([ele.Pinit for ele in self.slack_pre if ele.Bus in gen2attack])
            gf2droop = dPgen/Pgen_remain+1
            print(dPgen, Pgen_remain)
            self.contingency = {'type': c_type, 'location': gen2attack, 'parameter': gf2droop}
            print(self.contingency)
            # for ele in self.gen_pre:
            #     if ele.Bus in gen2attack:
            #         apdx = 'attack'
            #     else:
            #         apdx = ''
            #     print('create triton, gen ', ele.Bus, ele.P,apdx)
            # for ele in self.slack_pre:
            #     print('create triton, slack', ele.Bus, ele.Pinit)
        elif c_type=='substationoutage':
            ngen = len(self.gen_pre)
            K = 3  # attack K loads
            """choose K substations that will not cause islanding"""
            if_islanded = True #if the created contingency caused islanding?
            Max_trial = 10 #try at most ten times
            i_trial = 0
            while if_islanded and i_trial<Max_trial:
                i_trial = i_trial + 1
                slackbuses = [ele.Bus for ele in self.slack_pre]
                bus2attack = [ele.Bus for ele in self.bus_pre if ele.Bus not in slackbuses]
                bus2attack = np.random.choice(bus2attack,K) #any load or generation conected to buses are off
                #bus2attack = [28]
                print('substation outage at ', bus2attack)
                """check if network is islanded"""
                branch_edge = [(ele.from_bus,ele.to_bus)
                               for ele in self.branch_pre if ((ele.from_bus not in bus2attack) and (ele.to_bus not in bus2attack))]
                xfmr_edge = [(ele.from_bus,ele.to_bus)
                               for ele in self.xfmr_pre if ((ele.from_bus not in bus2attack) and (ele.to_bus not in bus2attack))]
                pairs = branch_edge + xfmr_edge
                graph = nx.from_edgelist(pairs)
                graph.add_nodes_from([ele.Bus for ele in self.bus_pre if ele.Bus not in bus2attack])
                islands = list(nx.connected_components(graph))
                if_islanded = len(islands)!=1
                if if_islanded:
                    print('network islanded, recreate contingency')
                else:
                    print('network not islanded, contingency created')
            """get the branches adjacent to the target substations"""
            branch2open = [[ele.from_bus, ele.to_bus]
                           for ele in self.branch_pre if ((ele.from_bus in bus2attack) or (ele.to_bus in bus2attack))]
            xfmr2open = [[ele.from_bus, ele.to_bus]
                         for ele in self.xfmr_pre if ((ele.from_bus in bus2attack) or (ele.to_bus in bus2attack))]
            """droop the generators if load or generation change"""
            buses_remain = islands[0]
            Pgen_remain = 0 + sum([ele.P for ele in self.gen_pre if ele.Bus in buses_remain]) + \
                          sum([ele.Pinit for ele in self.slack_pre if ele.Bus in buses_remain])
            dPgen = 0 + sum([ele.P for ele in self.gen_pre if ele.Bus not in buses_remain]) + \
                    sum([ele.Pinit for ele in self.slack_pre if ele.Bus not in buses_remain])
            dPload = 0 + sum([ele.P for ele in self.load_pre if ele.Bus not in buses_remain]) #load loss
            if dPgen-dPload == 0:
                gf2droop = 1
            else:
                gf2droop = (dPgen-dPload) / Pgen_remain + 1
            gen2attack = [ele.Bus for ele in self.gen_pre if ele.Bus in bus2attack]
            load2attack = [ele.Bus for ele in self.load_pre if ele.Bus in bus2attack]
            self.contingency = {'type': c_type, 'location': [bus2attack, gen2attack, load2attack, branch2open, xfmr2open], 'parameter': gf2droop}
            print(self.contingency, " affected load and gen:",dPgen,dPload)
            for ele in self.xfmr_pre:
                print("xfmr", ele.from_bus, ele.to_bus)


        
    def simulate(self, argvs, system = 'pre', method_name='file', enable_homotopy = False):
        #return power flow simulation of pre / post contingency system
        #argvs contains power flow simulation settings
        (MAXITER, PARSEROPTION, INITIALIZEOPTION, VOLTAGELIMIT, HOMOTOPYENABLE,
         VARIABLELIMITING, XHOMOTOPY, VOLTAGELIMITING, VrINIT, ViINIT, GENSETCONVERT,
         GB_MAN, G_CONT, B_CONT, IGCC, IGCC_NUM, VOLTAGE_LIMITING_GC, simID,
         CONTINGENCYON, QLIM, VOLTAGE_LIMITING_IMME_GC, ALLOW_DISCRETE_SS_CONTROL_AS_GEN, 
         ALLOW_SS_CONTROL, ENABLE_DISTRIBUTED_SLACK, ENABLE_DROOP, ENABLE_DROOP_HOMOTOPY, 
         STAMP_DUAL, 
         OBJECTIVE_MINIMIZE_FEASIBILITY_CURRENTS, LOAD_SHEDDING, UVLS, 
         ZBR_PARAM, SOLVER, REMOTEHIGHIMPEDANCE) = argvs
        if enable_homotopy:
            HOMOTOPYENABLE = True
        """----parsing contingency into manipulate -----"""
        manipulate = copy.deepcopy(self.manipulate) #self.manipulate.deepcopy()
        if system == 'post':
            #manipulate lf, gf, topology according to contingency
            print(self.contingency)
            #print(self.contingency['type']+' at location: ',self.contingency['location'], ', para',self.contingency['parameter'])
            if self.contingency['type'] == 'lineoutage':
                print('parsing lind contingency, convert to manipulate')
                manipulate['open branch'].extend(self.contingency['location'])
            elif self.contingency['type'] == 'MadIoT':
                print('parsing madiot contingency, convert to manipulate')
                # for ele in self.slack_pre:
                #     print('slack:',ele.Bus, ele.Pinit)
                # for ele in self.gen_pre:
                #     print('parsing madiot: gen  ', ele.Bus, ele.P)
                load2attack = self.contingency['location'] #bus names
                [lf2attack,gf2droop] = self.contingency['parameter'] #parameter
                manipulate['load scaling'].extend([[b, lf2attack] for b in load2attack])
                #print(np.array(manipulate['load scaling'], dtype=int)[:,0])
                manipulate['gf'] = manipulate['gf']*gf2droop
            elif self.contingency['type'] == 'genoutage':
                print('parsing gen contingency, convert to manipulate')
                manipulate['gen off'].extend(self.contingency['location'])
                manipulate['gf'] = manipulate['gf'] * self.contingency['parameter']
            elif self.contingency['type'] == 'substationoutage':
                print('parsing substation contingency, convert to manipulate')
                manipulate['bus off'].extend(self.contingency['location'][0])
                manipulate['gen off'].extend(self.contingency['location'][1])
                manipulate['load off'].extend(self.contingency['location'][2])
                manipulate['open branch'].extend(self.contingency['location'][3])
                manipulate['open xfmr'].extend(self.contingency['location'][4])
                manipulate['gf'] = manipulate['gf'] * self.contingency['parameter']
            else:
                raise
                #error('data.simulate not yet coded for manipulation in this way!')

        #gen/load outage contingency, input into runPF
        argvs = (self.case_name, manipulate, MAXITER, PARSEROPTION, INITIALIZEOPTION, VOLTAGELIMIT, HOMOTOPYENABLE,
         VARIABLELIMITING, XHOMOTOPY, VOLTAGELIMITING, VrINIT, ViINIT, GENSETCONVERT,
         GB_MAN, G_CONT, B_CONT, IGCC, IGCC_NUM, VOLTAGE_LIMITING_GC, simID,
         CONTINGENCYON, QLIM, VOLTAGE_LIMITING_IMME_GC, ALLOW_DISCRETE_SS_CONTROL_AS_GEN, 
         ALLOW_SS_CONTROL, ENABLE_DISTRIBUTED_SLACK, ENABLE_DROOP, ENABLE_DROOP_HOMOTOPY, 
         STAMP_DUAL, 
         OBJECTIVE_MINIMIZE_FEASIBILITY_CURRENTS, LOAD_SHEDDING, UVLS, 
         ZBR_PARAM, SOLVER, REMOTEHIGHIMPEDANCE)
        
        (bus, generator, slack, load, shunt, branch, xfmr,
         simID, flag_success, Niter,
         overlimit_lines, overlimit_xfmrs, overlimit_generators, max_adjoint_current) = runPF(argvs)
        #todo: bus.bus_name add to the class
        
        self.feasible = flag_success
        
        self.v[system] = dict()
        
        #print(system+'-contingency state variables:')
        for ele in bus:
            if ele.status:
                self.v[system][ele.Bus] = [ele.Vr_init, ele.Vi_init]
                #print(ele.name,ele.Vr_init, ele.Vi_init)

        update_bus_injection(bus, generator, slack, load, shunt)
        if system =='pre':
            self.bus_pre = bus
            self.load_pre = load
            self.gen_pre = generator
            self.slack_pre = slack
            self.branch_pre = branch
            self.xfmr_pre = xfmr
            #
            self.Niter['pre simu'] = Niter
            self.x['line feature'] = []
            self.x['xfmr feature'] = []
            self.x['bus feature'] = []
            self.x['shunt feature'] = []
            for ele in branch:
                if ele.status:
                    self.x['line feature'].append([ele.from_bus,ele.to_bus,ele.G_pu,ele.B_pu,ele.b]) #[busi,busj,g,b,bsh]
            for ele in xfmr:
                if ele.status:
                    zero_factor = 1
                    self.x['xfmr feature'].append([ele.from_bus,ele.to_bus,
                                                   ele.G_pu,ele.B_pu,ele.b,
                                                   ele.Gmag_raw, ele.Bmag_raw,
                                                   ele.tr, ele.ang, zero_factor]) #[busi,busj,g,b,bsh]
            for ele in bus:
                if ele.status:
                    self.x['bus feature'].append([ele.Bus, ele.Vr_init, ele.Vi_init, ele.Pinj, ele.Qinj, ele.Irinj, ele.Iiinj, ele.Qshunt]) #[busi, Pinj, Qinj] update_bus_injection(bus,generator,slack,load,shunt)
            for ele in shunt:
                self.x['shunt feature'].append([ele.Bus, None, ele.G_pu, ele.B_pu, 0 ]) #shunt can also be considered as an edge 
        elif system == 'post':
            self.Niter[method_name] = Niter
            dbus = dict() # net dPbus[busnum] = [dPgen, dPload, dQload]
            for ele in self.bus_pre:
                dbus[ele.Bus] = [0.0, 0.0, 0.0]
            #
            for ele in self.gen_pre:
                dbus[ele.Bus][0] = dbus[ele.Bus][0] - ele.P
            for ele in self.slack_pre:
                dbus[ele.Bus][0] = dbus[ele.Bus][0] - ele.Pinit
            for ele in generator:
                dbus[ele.Bus][0] = dbus[ele.Bus][0] + ele.P
            for ele in slack:
                dbus[ele.Bus][0] = 0.0 # theoretically we shouldn't know the actual change on slack bus becore simulating
                # dbus[ele.Bus][0] - ele.Pinit
            #
            for ele in self.load_pre:
                dbus[ele.Bus][1] = dbus[ele.Bus][1] - ele.P
                dbus[ele.Bus][2] = dbus[ele.Bus][2] - ele.Q
            for ele in load:
                dbus[ele.Bus][1] = dbus[ele.Bus][1] + ele.P
                dbus[ele.Bus][2] = dbus[ele.Bus][2] + ele.Q
            #
            for i, feat in enumerate(self.x['bus feature']):
                this_bus = feat[0]
                self.x['bus feature'][i].extend(dbus[this_bus])
                #print(this_bus, self.x['bus feature'][i])

            print("sample information ",self.case_name,)
            print("--pre: %d buses, %d loads, %d gen, %d lines, %d xfmrs"
                  %(len(self.bus_pre),  len(self.load_pre),len(self.gen_pre), len(self.branch_pre), len(self.xfmr_pre)))
            print(" --",self.contingency)
            # print(" --post: %d buses, %d loads, %d gen, %d lines, %d xfmrs"
            #       %(len(bus), len(load), len(generator), len(branch), len(xfmr)),flag_success)
            # for i,ele in enumerate(load):
            #     print(" ----load ", ele.Bus, ele.P, "pre", self.load_pre[i].Bus,self.load_pre[i].P)
            # for i,ele in enumerate(generator):
            #     print(" ----gen ", ele.Bus, ele.P, "pre", self.gen_pre[i].Bus,self.gen_pre[i].P)
            # for i, ele in enumerate(slack):
            #     print(" ----slack", ele.Bus, ele.Pinit, "pre", self.slack_pre[i].Pinit)
            self.bus_pre = []
            self.load_pre = []
            self.gen_pre = []
            self.slack_pre = []
            self.branch_pre = []
            self.xfmr_pre = []

