import numpy as np
from inertia_and_com import center_of_mass, inertia_tensor
from input_reader import hessian_reader, grad_reader

hess_file="o2.hess"

num_atoms, coord, cart_hess, M = hessian_reader(hess_file)

#def Align_to_principal(coord,mass):
    
def Identity(coord):
    """
    Parameters
    ----------
    coord : xyz coordinates, as read by hessian_reader 

    Returns
    -------
    The center-of-mass coordinates

    """
    com,mass=center_of_mass(coord)
    
    return np.matmul(np.eye(3),com.T)

def Inversion(coord):
    """
    Parameters
    ----------
    coord : xyz coordinates, as read by hessian_reader

    Returns
    -------
    The inverted center-of-mass xyz coordinates.

    """
    com,mass=center_of_mass(coord)
    return np.matmul(-np.eye(3),com.T)

def Reflection(coord):
    """
    Parameters
    ----------
    coord : xyz coordinates, as read by hessian_reader

    Returns
    -------
    The reflected center-of-mass coordinates: (refl_xy=[0], refl_xz=[1], refl_yz=[2])
    The reflection matrices: (sigma_xy=[3], sigma_xz=[4], sigma_yz=[5])
    """
    
    com,mass=center_of_mass(coord)    
    sigma_xy=np.array([[1,0,0],[0,1,0],[0,0,-1]])
    sigma_xz=np.array([[1,0,0],[0,-1,0],[0,0,1]])
    sigma_yz=np.array([[-1,0,0],[0,1,0],[0,0,1]])
    refl_xy=np.matmul(sigma_xy,com.T)
    refl_xz=np.matmul(sigma_xz,com.T)
    refl_yz=np.matmul(sigma_yz,com.T)
    return(refl_xy,refl_xz,refl_yz,sigma_xy,sigma_xz,sigma_yz)

def Rotation(coord,n):
    """
    Parameters
    ----------
    coord : xyz coordinates, as read by hessian_reader
    n : n-fold rotation, defines the rotation angle in radians 2pi/n. 
        for example: 2-fold -> 2pi/2: 180°, 3-fold -> 2pi/3: 120°, 4-fold -> 2pi/4: 90° 
    
    Returns
    -------
    The rotated center-of-mass coordinates: (rot_x=[0], rot_y=[1], rot_z=[2])
    The rotation matrices: (Rx=[3], Ry=[4], Rz=[5])
    """
    com,mass=center_of_mass(coord)
    angle = (2*np.pi)/n
    Rx=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    Ry=np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
    Rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    rot_x=np.matmul(Rx,com.T)
    rot_y=np.matmul(Ry,com.T)
    rot_z=np.matmul(Rz,com.T)
    return(rot_x,rot_y,rot_z,Rx,Ry,Rz)
