### calculates center-of-mass coordinates from input
import numpy as np

def center_of_mass(coord):
    x_com=y_com=z_com=mass=0
    mass_list=[]
    coord=np.delete(coord, [0], axis=1)
    coord=coord.astype(float)
    for i in range(len(coord)):
        x_com += coord[i][0]*coord[i][1]
        y_com += coord[i][0]*coord[i][2]
        z_com += coord[i][0]*coord[i][3]
        mass += coord[i][0]
        mass_list.append(coord[i][0])
    x_com=x_com/mass
    y_com=y_com/mass
    z_com=z_com/mass
    
    coord_com = np.zeros((len(coord),3))
    for i in range(len(coord)):
        coord_com[i,0] = coord[i][1]-x_com
        coord_com[i,1] = coord[i][2]-y_com
        coord_com[i,2] = coord[i][3]-z_com
    return (coord_com, mass_list)

def inertia_tensor(coord_com,mass):
    """
    Parameters
    ----------
    mass : amu masses of the atoms
        these can be retrieved from center_of_mass[1].
    coord_com : cartesian coordinates of the atoms
        The coordinates have been shifted such that the c.o.m
        corresponds to 0.0, 0.0, 0.0. This list can be retrieved from
        center_of_mass[0]

    Returns
    -------
    Ieigval : Principal moments of inertia, in amu/m2
        
    B : Rotational constants of the rigid-rotor
        in cm-1.
    """
    pi=np.pi
    c=29979245800 # cm/s
    bohr_to_meter=5.29177249E-11    # m/bohr
    h=6.62607015E-34 # Js
    NA=6.0221408E23 # 1/mol
    Ixx=Ixy=Ixz=Iyy=Iyz=Izz=0
    for i in range(len(coord_com)):
        Ixx=Ixx+mass[i]*(coord_com[i][1]**2+coord_com[i][2]**2)
        Iyy=Iyy+mass[i]*(coord_com[i][0]**2+coord_com[i][2]**2)
        Izz=Izz+mass[i]*(coord_com[i][0]**2+coord_com[i][1]**2)
        Ixy=Ixy-mass[i]*(coord_com[i][0]*coord_com[i][1])
        Ixz=Ixz-mass[i]*(coord_com[i][0]*coord_com[i][2])
        Iyz=Iyz-mass[i]*(coord_com[i][1]*coord_com[i][2])
    I_matrix = [[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]]
    Ieigval, Ieigvec = np.linalg.eigh(I_matrix)
    idx=Ieigval.argsort()
    Ieigval = Ieigval[idx]
    Ieigval=Ieigval*bohr_to_meter**2
    B=h/(8*pi**2*c*Ieigval)*1000*NA
    return (Ieigval, Ieigvec, B)