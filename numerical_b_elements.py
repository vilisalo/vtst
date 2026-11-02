import numpy as np

def vector(a,b):
    return np.subtract(a,b)

def length(a,b):
    return np.sqrt(np.dot(vector(a,b),vector(a,b)))

def angle(a,b,c):
    ab=vector(b,a)
    bc=vector(b,c)
    ab_len=length(b,a)
    bc_len=length(b,c)
    return np.arccos(np.dot(ab,bc)/(ab_len*bc_len))

def dihedral(a,b,c,d):
    ab=vector(a,b)
    bc=vector(b,c)
    cd=vector(c,d)
    cross_abc=np.cross(ab,bc)
    cross_bcd=np.cross(bc,cd)
    cross_abc_len=np.linalg.norm(cross_abc)
    cross_bcd_len=np.linalg.norm(cross_bcd)
    sign=np.sign(np.dot(cross_abc,cd))
    if not sign == 0:
        return -1*np.arccos(np.dot(cross_abc,cross_bcd)/(cross_abc_len*cross_bcd_len))
    return np.arccos(np.dot(cross_abc,cross_bcd)/(cross_abc_len*cross_bcd_len))


#pac=mol.principal_axis_coordinates
stepsize=1E-5
bmat=np.zeros((len(mol.int),3*len(mol.atoms)))
for j,i in enumerate(mol.int):
    idx1,idx2,idx3,idx4=i[0]-1,i[1]-1,i[2]-1,i[3]-1
    r0,r1,r2,r3=pac[idx1],pac[idx2],pac[idx3],pac[idx4]    
    
    if i[2]==0 and i[3]==0:	# Stretching coordinate
        bmat[j][3*idx1]  =(-length(r0+[2*stepsize,0,0],r1)+8*length(r0+[stepsize,0,0],r1)-8*length(r0-[stepsize,0,0],r1)+length(r0-[2*stepsize,0,0],r1))/(12*stepsize)
        bmat[j][3*idx1+1]=(-length(r0+[0,2*stepsize,0],r1)+8*length(r0+[0,stepsize,0],r1)-8*length(r0-[0,stepsize,0],r1)+length(r0-[0,2*stepsize,0],r1))/(12*stepsize)
        bmat[j][3*idx1+2]=(-length(r0+[0,0,2*stepsize],r1)+8*length(r0+[0,0,stepsize],r1)-8*length(r0-[0,0,stepsize],r1)+length(r0-[0,0,2*stepsize],r1))/(12*stepsize)
        bmat[j][3*idx2]  =(-length(r1+[2*stepsize,0,0],r0)+8*length(r1+[stepsize,0,0],r0)-8*length(r1-[stepsize,0,0],r0)+length(r1-[2*stepsize,0,0],r0))/(12*stepsize)
        bmat[j][3*idx2+1]=(-length(r1+[0,2*stepsize,0],r0)+8*length(r1+[0,stepsize,0],r0)-8*length(r1-[0,stepsize,0],r0)+length(r1-[0,2*stepsize,0],r0))/(12*stepsize)
        bmat[j][3*idx2+2]=(-length(r1+[0,0,2*stepsize],r0)+8*length(r1+[0,0,stepsize],r0)-8*length(r1-[0,0,stepsize],r0)+length(r1-[0,0,2*stepsize],r0))/(12*stepsize)

    elif i[2]!=0 and i[3]==0:	# Bending coordinate
        bmat[j][3*idx1]  =(-angle(r0+[2*stepsize,0,0],r1,r2)+8*angle(r0+[stepsize,0,0],r1,r2)-8*angle(r0-[stepsize,0,0],r1,r2)+angle(r0-[2*stepsize,0,0],r1,r2))/(12*stepsize)
        bmat[j][3*idx1+1]=(-angle(r0+[0,2*stepsize,0],r1,r2)+8*angle(r0+[0,stepsize,0],r1,r2)-8*angle(r0-[0,stepsize,0],r1,r2)+angle(r0-[0,2*stepsize,0],r1,r2))/(12*stepsize)
        bmat[j][3*idx1+2]=(-angle(r0+[0,0,2*stepsize],r1,r2)+8*angle(r0+[0,0,stepsize],r1,r2)-8*angle(r0-[0,0,stepsize],r1,r2)+angle(r0-[0,0,2*stepsize],r1,r2))/(12*stepsize)
        bmat[j][3*idx2]  =(-angle(r0,r1+[2*stepsize,0,0],r2)+8*angle(r0,r1+[stepsize,0,0],r2)-8*angle(r0,r1-[stepsize,0,0],r2)+angle(r0,r1-[2*stepsize,0,0],r2))/(12*stepsize)
        bmat[j][3*idx2+1]=(-angle(r0,r1+[0,2*stepsize,0],r2)+8*angle(r0,r1+[0,stepsize,0],r2)-8*angle(r0,r1-[0,stepsize,0],r2)+angle(r0,r1-[0,2*stepsize,0],r2))/(12*stepsize)
        bmat[j][3*idx2+2]=(-angle(r0,r1+[0,0,2*stepsize],r2)+8*angle(r0,r1+[0,0,stepsize],r2)-8*angle(r0,r1-[0,0,stepsize],r2)+angle(r0,r1-[0,0,2*stepsize],r2))/(12*stepsize)
        bmat[j][3*idx3]  =(-angle(r0,r1,r2+[2*stepsize,0,0])+8*angle(r0,r1,r2+[stepsize,0,0])-8*angle(r0,r1,r2-[stepsize,0,0])+angle(r0,r1,r2-[2*stepsize,0,0]))/(12*stepsize)
        bmat[j][3*idx3+1]=(-angle(r0,r1,r2+[0,2*stepsize,0])+8*angle(r0,r1,r2+[0,stepsize,0])-8*angle(r0,r1,r2-[0,stepsize,0])+angle(r0,r1,r2-[0,2*stepsize,0]))/(12*stepsize)
        bmat[j][3*idx3+2]=(-angle(r0,r1,r2+[0,0,2*stepsize])+8*angle(r0,r1,r2+[0,0,stepsize])-8*angle(r0,r1,r2-[0,0,stepsize])+angle(r0,r1,r2-[0,0,2*stepsize]))/(12*stepsize)

    else:				# Torsional or improper didhedral coordinate
        bmat[j][3*idx1]  =(-dihedral(r0+[2*stepsize,0,0],r1,r2,r3)+8*dihedral(r0+[stepsize,0,0],r1,r2,r3)-8*dihedral(r0-[stepsize,0,0],r1,r2,r3)+dihedral(r0-[2*stepsize,0,0],r1,r2,r3))/(12*stepsize)
        bmat[j][3*idx1+1]=(-dihedral(r0+[0,2*stepsize,0],r1,r2,r3)+8*dihedral(r0+[0,stepsize,0],r1,r2,r3)-8*dihedral(r0-[0,stepsize,0],r1,r2,r3)+dihedral(r0-[0,2*stepsize,0],r1,r2,r3))/(12*stepsize)
        bmat[j][3*idx1+2]=(-dihedral(r0+[0,0,2*stepsize],r1,r2,r3)+8*dihedral(r0+[0,0,stepsize],r1,r2,r3)-8*dihedral(r0-[0,0,stepsize],r1,r2,r3)+dihedral(r0-[0,0,2*stepsize],r1,r2,r3))/(12*stepsize)
        bmat[j][3*idx2]  =(-dihedral(r0,r1+[2*stepsize,0,0],r2,r3)+8*dihedral(r0,r1+[stepsize,0,0],r2,r3)-8*dihedral(r0,r1-[stepsize,0,0],r2,r3)+dihedral(r0,r1-[2*stepsize,0,0],r2,r3))/(12*stepsize)
        bmat[j][3*idx2+1]=(-dihedral(r0,r1+[0,2*stepsize,0],r2,r3)+8*dihedral(r0,r1+[0,stepsize,0],r2,r3)-8*dihedral(r0,r1-[0,stepsize,0],r2,r3)+dihedral(r0,r1-[0,2*stepsize,0],r2,r3))/(12*stepsize)
        bmat[j][3*idx2+2]=(-dihedral(r0,r1+[0,0,2*stepsize],r2,r3)+8*dihedral(r0,r1+[0,0,stepsize],r2,r3)-8*dihedral(r0,r1-[0,0,stepsize],r2,r3)+dihedral(r0,r1-[0,0,2*stepsize],r2,r3))/(12*stepsize)
        bmat[j][3*idx3]  =(-dihedral(r0,r1,r2+[2*stepsize,0,0],r3)+8*dihedral(r0,r1,r2+[stepsize,0,0],r3)-8*dihedral(r0,r1,r2-[stepsize,0,0],r3)+dihedral(r0,r1,r2-[2*stepsize,0,0],r3))/(12*stepsize)
        bmat[j][3*idx3+1]=(-dihedral(r0,r1,r2+[0,2*stepsize,0],r3)+8*dihedral(r0,r1,r2+[0,stepsize,0],r3)-8*dihedral(r0,r1,r2-[0,stepsize,0],r3)+dihedral(r0,r1,r2-[0,2*stepsize,0],r3))/(12*stepsize)
        bmat[j][3*idx3+2]=(-dihedral(r0,r1,r2+[0,0,2*stepsize],r3)+8*dihedral(r0,r1,r2+[0,0,stepsize],r3)-8*dihedral(r0,r1,r2-[0,0,stepsize],r3)+dihedral(r0,r1,r2-[0,0,2*stepsize],r3))/(12*stepsize)
        bmat[j][3*idx4]  =(-dihedral(r0,r1,r2,r3+[2*stepsize,0,0])+8*dihedral(r0,r1,r2,r3+[stepsize,0,0])-8*dihedral(r0,r1,r2,r3-[stepsize,0,0])+dihedral(r0,r1,r2,r3-[2*stepsize,0,0]))/(12*stepsize)
        bmat[j][3*idx4+1]=(-dihedral(r0,r1,r2,r3+[0,2*stepsize,0])+8*dihedral(r0,r1,r2,r3+[0,stepsize,0])-8*dihedral(r0,r1,r2,r3-[0,stepsize,0])+dihedral(r0,r1,r2,r3-[0,2*stepsize,0]))/(12*stepsize)
        bmat[j][3*idx4+2]=(-dihedral(r0,r1,r2,r3+[0,0,2*stepsize])+8*dihedral(r0,r1,r2,r3+[0,0,stepsize])-8*dihedral(r0,r1,r2,r3-[0,0,stepsize])+dihedral(r0,r1,r2,r3-[0,0,2*stepsize]))/(12*stepsize)
