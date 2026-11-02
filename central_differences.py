import numpy as np

def central_diff_length(x,idx1,idx2,h=1e-5):
    def length(components):
        p0,p1=components[3*idx1:3*idx1+3],components[3*idx2:3*idx2+3]
        return np.sqrt(np.dot(np.subtract(p0,p1),np.subtract(p0,p1)))
    gradient = np.zeros_like(x)
    for i in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2:
        xph = x.copy()
        xp2h = x.copy()
        xmh = x.copy() 
        xm2h = x.copy() 
        xph[i] += h
        xp2h[i] += 2*h
        xmh[i] -= h
        xm2h[i] -= 2*h        
        len_xp2h=length(xp2h)
        len_xph=length(xph)
        len_xmh=length(xmh)
        len_xm2h=length(xm2h)
        gradient[i] = (-len_xp2h + 8*len_xph - 8*len_xmh + len_xm2h)/(12*h)
    return gradient

def central_diff_angle(x,idx1,idx2,idx3,h=1e-5):
    def angle(components):
        p0,p1,p2=components[3*idx1:3*idx1+3],components[3*idx2:3*idx2+3],components[3*idx3:3*idx3+3]
        ab=np.subtract(p1,p0)
        bc=np.subtract(p1,p2)
        len_ab = np.sqrt(np.dot(ab,ab)) 
        len_bc = np.sqrt(np.dot(bc,bc))
        return np.arccos(np.dot(ab,bc)/(len_ab*len_bc))        
    gradient = np.zeros_like(x)
    for i in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2,3*idx3,3*idx3+1,3*idx3+2:
        xph = x.copy()
        xp2h = x.copy()
        xmh = x.copy() 
        xm2h = x.copy() 
        xph[i] += h
        xp2h[i] += 2*h
        xmh[i] -= h
        xm2h[i] -= 2*h        
        ang_xp2h=angle(xp2h)
        ang_xph=angle(xph)
        ang_xmh=angle(xmh)
        ang_xm2h=angle(xm2h)
        gradient[i] = (-ang_xp2h + 8*ang_xph - 8*ang_xmh + ang_xm2h)/(12*h)
    return gradient        

def central_diff_dihed(x,idx1,idx2,idx3,idx4,h=1e-5):
    def dihedral(components):
        p0,p1,p2,p3=components[3*idx1:3*idx1+3],components[3*idx2:3*idx2+3],components[3*idx3:3*idx3+3],components[3*idx4:3*idx4+3]
        ab=np.subtract(p0,p1)
        bc=np.subtract(p1,p2)
        cd=np.subtract(p2,p3)
        cross_abc=np.cross(ab,bc)
        cross_bcd=np.cross(bc,cd)
        cross_abc_len=np.linalg.norm(cross_abc)
        cross_bcd_len=np.linalg.norm(cross_bcd)
        sign=np.sign(np.dot(cross_abc,cd))
        if not sign == 0:
            return -1*np.arccos(np.dot(cross_abc,cross_bcd)/(cross_abc_len*cross_bcd_len))
        return np.arccos(np.dot(cross_abc,cross_bcd)/(cross_abc_len*cross_bcd_len))
    gradient = np.zeros_like(x)
    for i in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2,3*idx3,3*idx3+1,3*idx3+2,3*idx4,3*idx4+1,3*idx4+2:
        xph = x.copy()
        xp2h = x.copy()
        xmh = x.copy() 
        xm2h = x.copy() 
        xph[i] += h
        xp2h[i] += 2*h
        xmh[i] -= h
        xm2h[i] -= 2*h        
        dih_xp2h=dihedral(xp2h)
        dih_xph=dihedral(xph)
        dih_xmh=dihedral(xmh)
        dih_xm2h=dihedral(xm2h)
        gradient[i] = (-dih_xp2h + 8*dih_xph - 8*dih_xmh + dih_xm2h)/(12*h)
    return gradient

def central_diff_length_2nd(x,idx1,idx2,h=1e-5):
    def length(components):
        p0,p1=components[3*idx1:3*idx1+3],components[3*idx2:3*idx2+3]
        return np.sqrt(np.dot(np.subtract(p0,p1),np.subtract(p0,p1)))        
    hessian = np.zeros((len(x),len(x)))
    for i in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2:
        for j in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2:
            xp2h_yp2h = x.copy()
            xph_yph = x.copy()
            xp2h_ym2h = x.copy()
            xph_ymh = x.copy()
            xm2h_yp2h = x.copy()
            xmh_yph = x.copy()
            xm2h_ym2h = x.copy()
            xmh_ymh = x.copy()
            xp2h_yp2h[i] += 2*h
            xp2h_yp2h[j] += 2*h
            xph_yph[i] += h
            xph_yph[j] += h
            xp2h_ym2h[i] += 2*h
            xp2h_ym2h[j] -= 2*h
            xph_ymh[i] += h
            xph_ymh[j] -= h
            xm2h_yp2h[i] -= 2*h
            xm2h_yp2h[j] += 2*h
            xmh_yph[i] -= h
            xmh_yph[j] += h
            xm2h_ym2h[i] -= 2*h
            xm2h_ym2h[j] -= 2*h
            xmh_ymh[i] -= h
            xmh_ymh[j] -= h
            len_xp2h_yp2h=length(xp2h_yp2h)
            len_xph_yph=length(xph_yph)
            len_xp2h_ym2h=length(xp2h_ym2h)
            len_xph_ymh=length(xph_ymh)
            len_xm2h_yp2h=length(xm2h_yp2h)
            len_xmh_yph=length(xmh_yph)
            len_xm2h_ym2h=length(xm2h_ym2h)
            len_xmh_ymh=length(xmh_ymh)
            hessian[i][j] = (-len_xp2h_yp2h + 16*len_xph_yph + len_xp2h_ym2h - 16*len_xph_ymh + len_xm2h_yp2h - 16*len_xmh_yph - len_xm2h_ym2h + 16*len_xmh_ymh)/(48*h**2)
    return hessian

def central_diff_angle_2nd(x,idx1,idx2,idx3,h=1e-5):
    def angle(components):
        p0,p1,p2=components[3*idx1:3*idx1+3],components[3*idx2:3*idx2+3],components[3*idx3:3*idx3+3]
        ab=np.subtract(p1,p0)
        bc=np.subtract(p1,p2)
        len_ab = np.sqrt(np.dot(ab,ab)) 
        len_bc = np.sqrt(np.dot(bc,bc))
        return np.arccos(np.dot(ab,bc)/(len_ab*len_bc))        
    hessian = np.zeros((len(x),len(x)))
    for i in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2,3*idx3,3*idx3+1,3*idx3+2:
        for j in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2,3*idx3,3*idx3+1,3*idx3+2:
            xp2h_yp2h = x.copy()
            xph_yph = x.copy()
            xp2h_ym2h = x.copy()
            xph_ymh = x.copy()
            xm2h_yp2h = x.copy()
            xmh_yph = x.copy()
            xm2h_ym2h = x.copy()
            xmh_ymh = x.copy()
            xp2h_yp2h[i] += 2*h
            xp2h_yp2h[j] += 2*h
            xph_yph[i] += h
            xph_yph[j] += h
            xp2h_ym2h[i] += 2*h
            xp2h_ym2h[j] -= 2*h
            xph_ymh[i] += h
            xph_ymh[j] -= h
            xm2h_yp2h[i] -= 2*h
            xm2h_yp2h[j] += 2*h
            xmh_yph[i] -= h
            xmh_yph[j] += h
            xm2h_ym2h[i] -= 2*h
            xm2h_ym2h[j] -= 2*h
            xmh_ymh[i] -= h
            xmh_ymh[j] -= h
            ang_xp2h_yp2h=angle(xp2h_yp2h)
            ang_xph_yph=angle(xph_yph)
            ang_xp2h_ym2h=angle(xp2h_ym2h)
            ang_xph_ymh=angle(xph_ymh)
            ang_xm2h_yp2h=angle(xm2h_yp2h)
            ang_xmh_yph=angle(xmh_yph)
            ang_xm2h_ym2h=angle(xm2h_ym2h)
            ang_xmh_ymh=angle(xmh_ymh)
            hessian[i][j] = (-ang_xp2h_yp2h + 16*ang_xph_yph + ang_xp2h_ym2h - 16*ang_xph_ymh + ang_xm2h_yp2h - 16*ang_xmh_yph - ang_xm2h_ym2h + 16*ang_xmh_ymh)/(48*h**2)
    return hessian 

def central_diff_dihed_2nd(x,idx1,idx2,idx3,idx4,h=1e-5):
    def dihedral(components):
        p0,p1,p2,p3=components[3*idx1:3*idx1+3],components[3*idx2:3*idx2+3],components[3*idx3:3*idx3+3],components[3*idx4:3*idx4+3]
        ab=np.subtract(p0,p1)
        bc=np.subtract(p1,p2)
        cd=np.subtract(p2,p3)
        cross_abc=np.cross(ab,bc)
        cross_bcd=np.cross(bc,cd)
        cross_abc_len=np.linalg.norm(cross_abc)
        cross_bcd_len=np.linalg.norm(cross_bcd)
        sign=np.sign(np.dot(cross_abc,cd))
        if not sign == 0:
            return -1*np.arccos(np.dot(cross_abc,cross_bcd)/(cross_abc_len*cross_bcd_len))
        return np.arccos(np.dot(cross_abc,cross_bcd)/(cross_abc_len*cross_bcd_len))
    hessian=np.zeros((len(x),len(x)))
    for i in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2,3*idx3,3*idx3+1,3*idx3+2,3*idx4,3*idx4+1,3*idx4+2:
        for j in 3*idx1,3*idx1+1,3*idx1+2,3*idx2,3*idx2+1,3*idx2+2,3*idx3,3*idx3+1,3*idx3+2,3*idx4,3*idx4+1,3*idx4+2:
            xp2h_yp2h = x.copy()
            xph_yph = x.copy()
            xp2h_ym2h = x.copy()
            xph_ymh = x.copy()
            xm2h_yp2h = x.copy()
            xmh_yph = x.copy()
            xm2h_ym2h = x.copy()
            xmh_ymh = x.copy()
            xp2h_yp2h[i] += 2*h
            xp2h_yp2h[j] += 2*h
            xph_yph[i] += h
            xph_yph[j] += h
            xp2h_ym2h[i] += 2*h
            xp2h_ym2h[j] -= 2*h
            xph_ymh[i] += h
            xph_ymh[j] -= h
            xm2h_yp2h[i] -= 2*h
            xm2h_yp2h[j] += 2*h
            xmh_yph[i] -= h
            xmh_yph[j] += h
            xm2h_ym2h[i] -= 2*h
            xm2h_ym2h[j] -= 2*h
            xmh_ymh[i] -= h
            xmh_ymh[j] -= h
            dih_xp2h_yp2h=dihedral(xp2h_yp2h)
            dih_xph_yph=dihedral(xph_yph)
            dih_xp2h_ym2h=dihedral(xp2h_ym2h)
            dih_xph_ymh=dihedral(xph_ymh)
            dih_xm2h_yp2h=dihedral(xm2h_yp2h)
            dih_xmh_yph=dihedral(xmh_yph)
            dih_xm2h_ym2h=dihedral(xm2h_ym2h)
            dih_xmh_ymh=dihedral(xmh_ymh)
            hessian[i][j] = (-dih_xp2h_yp2h + 16*dih_xph_yph + dih_xp2h_ym2h - 16*dih_xph_ymh + dih_xm2h_yp2h - 16*dih_xmh_yph - dih_xm2h_ym2h + 16*dih_xmh_ymh)/(48*h**2)
    return hessian
      
        
def get_bmat(cart_coord,int_coord):
    idx1,idx2,idx3,idx4=int_coord[0]-1,int_coord[1]-1,int_coord[2]-1,int_coord[3]-1
    bmat_row=np.zeros_like(cart_coord)
    if int_coord[2]==0 and int_coord[3]==0:     #stretching coordinates
        bmat_row[3*idx1:3*idx1+3]=cart_coord[3*idx1:3*idx1+3]
        bmat_row[3*idx2:3*idx2+3]=cart_coord[3*idx2:3*idx2+3]
        gradient=central_diff_length(bmat_row,idx1,idx2)
        return gradient
    elif int_coord[2]!=0 and int_coord[3]==0:   #angle coordinates
        bmat_row[3*idx1:3*idx1+3]=cart_coord[3*idx1:3*idx1+3]
        bmat_row[3*idx2:3*idx2+3]=cart_coord[3*idx2:3*idx2+3]
        bmat_row[3*idx3:3*idx3+3]=cart_coord[3*idx3:3*idx3+3]
        gradient=central_diff_angle(bmat_row,idx1,idx2,idx3)
        return gradient
    else:                                       #dihedral coordinates
        bmat_row[3*idx1:3*idx1+3]=cart_coord[3*idx1:3*idx1+3]
        bmat_row[3*idx2:3*idx2+3]=cart_coord[3*idx2:3*idx2+3]
        bmat_row[3*idx3:3*idx3+3]=cart_coord[3*idx3:3*idx3+3]
        bmat_row[3*idx4:3*idx4+3]=cart_coord[3*idx4:3*idx4+3]
        gradient=central_diff_dihed(bmat_row,idx1,idx2,idx3,idx4)
        return gradient

def get_cmat(cart_coord,int_coord):
    idx1,idx2,idx3,idx4=int_coord[0]-1,int_coord[1]-1,int_coord[2]-1,int_coord[3]-1
    bmat_row=np.zeros_like(cart_coord)
    if int_coord[2]==0 and int_coord[3]==0:     #stretching coordinates
        bmat_row[3*idx1:3*idx1+3]=cart_coord[3*idx1:3*idx1+3]
        bmat_row[3*idx2:3*idx2+3]=cart_coord[3*idx2:3*idx2+3]
        hessian=central_diff_length_2nd(bmat_row,idx1,idx2)
        return hessian
    elif int_coord[2]!=0 and int_coord[3]==0:   #angle coordinates
        bmat_row[3*idx1:3*idx1+3]=cart_coord[3*idx1:3*idx1+3]
        bmat_row[3*idx2:3*idx2+3]=cart_coord[3*idx2:3*idx2+3]
        bmat_row[3*idx3:3*idx3+3]=cart_coord[3*idx3:3*idx3+3]
        hessian=central_diff_angle_2nd(bmat_row,idx1,idx2,idx3)
        return hessian
    else:                                       #dihedral coordinates
        bmat_row[3*idx1:3*idx1+3]=cart_coord[3*idx1:3*idx1+3]
        bmat_row[3*idx2:3*idx2+3]=cart_coord[3*idx2:3*idx2+3]
        bmat_row[3*idx3:3*idx3+3]=cart_coord[3*idx3:3*idx3+3]
        bmat_row[3*idx4:3*idx4+3]=cart_coord[3*idx4:3*idx4+3]
        hessian=central_diff_dihed_2nd(bmat_row,idx1,idx2,idx3,idx4)
        return hessian
        
    
    
    
coord=np.array([[-0.22145133,  0.00003968, -0.29502026],
           [-0.37963275, -0.00302923,  3.16841498],
           [ 0.58745916, -1.66800456,  3.89740706],
           [ 0.58846449,  1.66000346,  3.9005081 ],
           [-2.32648719, -0.00306892,  3.84870315],
           [-2.30295821,  2.67312721, -1.03473844],
           [-1.40272481,  4.42832184, -0.44514578],
           [-2.60874991,  2.76914041, -3.06855996],
           [-4.12841285,  2.50505686, -0.08984325],
           [-2.30502557, -2.67016223, -1.03948165],
           [-2.61123868, -2.76204449, -3.07342789],
           [-1.40599025, -4.42712753, -0.45335286],
           [-4.13019865, -2.5025284 , -0.09396474],
           [ 5.82953864,  0.00012094, -0.67037089],
           [ 4.04902924, -0.00094297, -1.19922398]])