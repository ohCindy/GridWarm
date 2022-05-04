from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse.linalg import spsolve


def stamp_matrix(i,j,v,row,col,val):
    row.append(i)
    col.append(j)
    val.append(v)
    return row, col, val

def stampVCCS(g1,g2,vk,vl,v, row, col, val):
    row.append(g1)
    col.append(vk)
    val.append(v)
    #
    row.append(g1)
    col.append(vl)
    val.append(-v)
    #
    row.append(g2)
    col.append(vl)
    val.append(v)
    #
    row.append(g2)
    col.append(vk)
    val.append(-v)
    return row, col, val

def stamp_line(fR_node,fI_node, tR_node, tI_node, G,B,bsh, row,col,val):
    # stamp B
    row, col, val = stamp_matrix(fR_node, fI_node, -B, row, col, val)
    row, col, val = stamp_matrix(fR_node, tI_node, B, row, col, val)
    row, col, val = stamp_matrix(fI_node, fR_node, B, row, col, val)
    row, col, val = stamp_matrix(fI_node, tR_node, -B, row, col, val)
    #
    row, col, val = stamp_matrix(tR_node, fI_node, B, row, col, val)
    row, col, val = stamp_matrix(tR_node, tI_node, -B, row, col, val)
    row, col, val = stamp_matrix(tI_node, fR_node, -B, row, col, val)
    row, col, val = stamp_matrix(tI_node, tR_node, B, row, col, val)
    # stamp G
    if not (G == 0):
        row, col, val = stamp_matrix(fR_node, fR_node, G, row, col, val)
        row, col, val = stamp_matrix(fI_node, fI_node, G, row, col, val)
        row, col, val = stamp_matrix(tR_node, tR_node, G, row, col, val)
        row, col, val = stamp_matrix(tI_node, tI_node, G, row, col, val)
        #
        row, col, val = stamp_matrix(fR_node, tR_node, -G, row, col, val)
        row, col, val = stamp_matrix(fI_node, tI_node, -G, row, col, val)
        row, col, val = stamp_matrix(tR_node, fR_node, -G, row, col, val)
        row, col, val = stamp_matrix(tI_node, fI_node, -G, row, col, val)
    # stamp shunt bsh
    row, col, val = stamp_matrix(fR_node, fI_node, -0.5 * bsh, row, col, val)
    row, col, val = stamp_matrix(fI_node, fR_node, 0.5 * bsh, row, col, val)
    row, col, val = stamp_matrix(tR_node, tI_node, -0.5 * bsh, row, col, val)
    row, col, val = stamp_matrix(tI_node, tR_node, 0.5 * bsh, row, col, val)
    return row,col,val

def stamp_xfmr(fR_node,fI_node, tR_node, tI_node,
               G_pu,B_pu,bsh, Gmag_raw, Bmag_raw,
               tr,  ang, zero_factor,
               row,col,val):
    tr2 = tr * tr
    if ang == 0:
        #zero angle
        G_series = zero_factor * G_pu / tr
        B_series = zero_factor * B_pu / tr
        G_shunt_from = zero_factor * (1 - tr) / (tr2) * G_pu
        B_shunt_from = zero_factor * ((1 - tr) / (tr2) * B_pu + (bsh / 2) / tr2)
        G_shunt_to = zero_factor * G_pu * (1 - 1 / tr)
        B_shunt_to = zero_factor * ((1 - 1 / tr) * B_pu + bsh / 2)
        if G_pu:
            #real
            row, col, val = stamp_matrix(fR_node, fR_node, G_shunt_from, row, col, val)
            row, col, val = stamp_matrix(fR_node, fR_node, G_series, row, col, val)
            row, col, val = stamp_matrix(fR_node, tR_node, -G_series, row, col, val)
            row, col, val = stamp_matrix(tR_node, tR_node, G_series, row, col, val)
            row, col, val = stamp_matrix(tR_node, fR_node, -G_series, row, col, val)
            row, col, val = stamp_matrix(tR_node, tR_node, G_shunt_to, row, col, val)
            # imaginary
            row, col, val = stamp_matrix(fI_node, fI_node, G_shunt_from, row, col, val)
            row, col, val = stamp_matrix(fI_node, fI_node, G_series, row, col, val)
            row, col, val = stamp_matrix(fI_node, tI_node, -G_series, row, col, val)
            row, col, val = stamp_matrix(tI_node, tI_node, G_series, row, col, val)
            row, col, val = stamp_matrix(tI_node, fI_node, -G_series, row, col, val)
            row, col, val = stamp_matrix(tI_node, tI_node, G_shunt_to, row, col, val)
        row, col, val = stampVCCS(fR_node, tR_node, fI_node, tI_node,-B_series, row, col, val)
        row, col, val = stamp_matrix(fR_node, fI_node, -B_shunt_from, row, col, val)
        row, col, val = stamp_matrix(tR_node, tI_node, -B_shunt_to, row, col, val)
        row, col, val = stampVCCS(fI_node, tI_node, fR_node, tR_node,
                                  B_series, row, col, val)
        row, col, val = stamp_matrix(fI_node, fR_node, B_shunt_from, row, col, val)
        row, col, val = stamp_matrix(tI_node, tR_node, B_shunt_to, row, col, val)
    else:
        Gt = G_pu
        Bt = B_pu + bsh / 2
        phi_rad = ang * np.pi / 180
        cosphi = np.cos(phi_rad)
        sinphi = np.sin(phi_rad)
        Gcosphi = G_pu * cosphi
        Gsinphi = G_pu * sinphi
        Bcosphi = B_pu * cosphi
        Bsinphi = B_pu * sinphi
        G_shunt_from = zero_factor * Gt / tr2
        B_shunt_from = zero_factor * Bt / tr2
        MR_from = zero_factor * (Gcosphi - Bsinphi) / tr
        MI_from = zero_factor * (Gsinphi + Bcosphi) / tr
        G_to = zero_factor * (Gcosphi + Bsinphi) / tr
        B_to = zero_factor * (Bcosphi - Gsinphi) / tr
        MR_to = zero_factor * Gt
        MI_to = zero_factor * Bt
        #stamp real
        row, col, val = stamp_matrix(fR_node, fR_node, G_shunt_from, row, col, val)
        row, col, val = stamp_matrix(fR_node, tR_node, -MR_from, row, col, val)
        row, col, val = stamp_matrix(fR_node, fI_node, -B_shunt_from, row, col, val)
        row, col, val = stamp_matrix(fR_node, tI_node, MI_from, row, col, val)
        #
        row, col, val = stamp_matrix(tR_node, tR_node, MR_to, row, col, val)
        row, col, val = stamp_matrix(tR_node, fR_node, -G_to, row, col, val)
        row, col, val = stamp_matrix(tR_node, fI_node, B_to, row, col, val)
        row, col, val = stamp_matrix(tR_node, tI_node, -MI_to, row, col, val)
        # imaginary part of the circuit
        row, col, val = stamp_matrix(fI_node, fI_node, G_shunt_from, row, col, val)
        row, col, val = stamp_matrix(fI_node, fR_node, B_shunt_from, row, col, val)
        row, col, val = stamp_matrix(fI_node, tR_node, -MI_from, row, col, val)
        row, col, val = stamp_matrix(fI_node, tI_node, -MR_from, row, col, val)
        #
        row, col, val = stamp_matrix(tI_node, tI_node, MR_to, row, col, val)
        row, col, val = stamp_matrix(tI_node, fR_node, -B_to, row, col, val)
        row, col, val = stamp_matrix(tI_node, fI_node, -G_to, row, col, val)
        row, col, val = stamp_matrix(tI_node, tR_node, MI_to, row, col, val)
    if Bmag_raw:
        row, col, val = stamp_matrix(fR_node, fI_node, -Bmag_raw, row, col, val)
        row, col, val = stamp_matrix(fI_node, fR_node, Bmag_raw, row, col, val)
    if Gmag_raw:
        row, col, val = stamp_matrix(fR_node, fR_node, Gmag_raw, row, col, val)
        row, col, val = stamp_matrix(fI_node, fI_node, Gmag_raw, row, col, val)
    return row, col, val


def makeYbus(bus_key_dict, Nbus, lines, xfmrs, shunts=[]):
    #branch is the branch features in generated data [frombusnum,tobusnum,G,B,b]
    # when calculating I=Yv, Yv gets the sum of outflow currents
    row = []
    col = []
    val = []
    sizeY = Nbus*2

    for ele in lines:
        fR_node = bus_key_dict[ele[0]][0]
        fI_node = bus_key_dict[ele[0]][1]
        tR_node = bus_key_dict[ele[1]][0]
        tI_node = bus_key_dict[ele[1]][1]
        G = ele[2]
        B = ele[3]
        bsh = ele[4]
        row,col,val = stamp_line(fR_node, fI_node, tR_node, tI_node,
                                 G, B, bsh, row, col, val)
    #
    for ele in xfmrs:
        fR_node = bus_key_dict[ele[0]][0]
        fI_node = bus_key_dict[ele[0]][1]
        tR_node = bus_key_dict[ele[1]][0]
        tI_node = bus_key_dict[ele[1]][1]
        G_pu = ele[2]
        B_pu = ele[3]
        bsh = ele[4]
        Gmag_raw = ele[5]
        Bmag_raw = ele[6]
        tr =  ele[7] #tranformation ratio
        ang = ele[8] #angle
        zero_factor = ele[9]
        #
        row, col, val = stamp_xfmr(fR_node,fI_node, tR_node, tI_node,
                                   G_pu,B_pu,bsh, Gmag_raw, Bmag_raw,
                                   tr,  ang, zero_factor,
                                   row,col,val)


    #
    for ele in shunts:
        #ele = [bus,None,G,B,0]
        R_node = bus_key_dict[ele[0]][0]
        I_node = bus_key_dict[ele[0]][1]
        G = ele[2]
        B = ele[3]
        if not (G==0):
            row, col, val = stamp_matrix(R_node, R_node, G, row, col, val)
            row, col, val = stamp_matrix(I_node, I_node, G, row, col, val)
        #stamp B
        row, col, val = stamp_matrix(R_node, I_node, -B, row, col, val)
        row, col, val = stamp_matrix(I_node, R_node, B, row, col, val)
    Ybus = csr_matrix((val, (row, col)), shape=(sizeY, sizeY))
    return Ybus
def makeJ(bus_key_dict, Nbus, bus):
    #bus is the bus features in generated data [busnum,Vr, Vi, Pinj, Qinj, Irinj, Iiinj, Qshunt]
    #in input, the injection direction is from node to ground
    #output J direction: from ground to node
    row = []
    col = []
    val = []
    for ele in bus:
        R_node = bus_key_dict[ele[0]][0]
        I_node = bus_key_dict[ele[0]][1]
        Ir_inj = ele[4]
        Ii_inj = ele[5]
        row, col, val = stamp_matrix(R_node, 0, -Ir_inj, row, col, val)
        row, col, val = stamp_matrix(I_node, 0, -Ii_inj, row, col, val)
    sizeJ = 2*Nbus
    J = csr_matrix((val, (row, col)), shape=(sizeJ, 1))
    return J

class LinearSystem():
    def __init__(self,bus,branch,shunts=[]):
        self.bus = bus #bus features in generated data, [busnum,pinj,qinj,Irinj,Iiinj]
        self.branch = branch #branch features in generated data [fromnum,tonum, G,B,bsh]
        self.shunts = shunts
        # if shunts:
        #     print('todo: need to modify the code with shunts included')
        self.make_utils()
    def make_utils(self):
        #make self.bus_key_dict: dict[busnumber]=[Rnode,Inode]
        #self.Nbus
        self.bus_key_dict = dict()
        key_count = 0  # count the number of node ids in matrix creation
        for ele in self.bus:
            self.bus_key_dict[ele[0]] = [key_count, key_count + 1]
            key_count = key_count + 2
        self.Nbus = len(self.bus)
    def update_case(self,contingency):
        #contingency is a dictionary, having keys: 'type','location', and 'parameter'
        #update self.bus,self.bus_key_dict and self.branch, according to contingency
        """Note that we typically do not change the size of Y and J,
        even if some buses are off after contingency,
        by doing this we make an (sensitivity) estimate for all buses in the pre-contingency layout"""
        outage_br = []
        outage_bus = []
        if contingency['type'] == 'line outage':
            outage_br = contingency['location']
        elif contingency['type'] == 'substation outage':
            error("todo")
        self.branch = [ele for ele in self.branch if not ([ele[0],ele[1]] in outage_br)]
        #update Ybus, J and dicts
        _ = self.makeYbus()
        _ = self.makeJ()
        if self.Nbus!=len(self.bus):
            self.make_utils()

    def makeYbus(self):
        Ybus = makeYbus(self.bus_key_dict, self.Nbus, self.branch, self.shunts)
        self.Ybus = Ybus
        return Ybus

    def makeJ(self):
        #make J from node features, feature=[busnum, Vr, Vi, Pinj, Qinj, Irinj, Iiinj, Qshunt,...]
        J = makeJ(self.bus_key_dict, self.Nbus, self.bus)
        self.J = J
        return J

    def getJ_from_v(self, v_dict):
        #input: v_dict: a dictionary v_dict[bus_num] = [vr,vi]
        #output: Injection dictionary: Inj_dict[bus num] = [Irinj,Iiinj]
        Inj_dict = v_dict.copy()
        if set(v_dict.keys())==set(self.bus_key_dict.keys()):
            v = np.array([v_dict[busnum] if len(v_dict[busnum])==2 else [0,0] for busnum in self.bus_key_dict.keys()]).reshape(-1, 1)  # nd array
        else:
            print('number of buses changed after contingency')
            v = []
            for busnum in list(self.bus_key_dict.keys()):
                if (busnum in list(v_dict.keys())) and (len(v_dict[busnum])==2):
                    v.append(v_dict[busnum])
                else:
                    v.append([0,0])
            v = np.array(v).reshape(-1,1)
        #
        try:
            J_calc = self.Ybus.dot(v)  # dense Nd array
        except:
            _ = self.makeYbus()
            J_calc = self.Ybus.dot(v)
        J_calc = J_calc.reshape(-1,2) #Nbus by 2 array
        for i,busnum in enumerate(self.bus_key_dict.keys()):
            if busnum in list(v_dict.keys()):
                Inj_dict[busnum] = J_calc[i]
        return Inj_dict
    def solvev_from_YJ(self):
        #solve v from Yv=J,
        #output Y is a
        try:
            vsol = spsolve(self.Ybus, self.J).reshape(-1,2)
        except:
            _ = self.makeYbus()
            _ = self.makeJ()
            vsol = spsolve(self.Ybus, self.J).reshape(-1, 2)
        v_dict = dict()
        for i, busnum in enumerate(self.bus_key_dict.keys()):
            v_dict[busnum] = vsol[i]
        return v_dict



