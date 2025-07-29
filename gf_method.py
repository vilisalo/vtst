import numpy as np
import output
import tools

def vector(A, B):
    vector = np.subtract(A,B)
    return vector

# Length of vector connecting points A and B
def length(A, B):
    length = np.sqrt(np.dot(vector(A,B), vector(A,B)))
    return length

def evector(A, B):
    evector = vector(A,B)/length(A,B)
    return evector

# Stretching

def bonds_svectors(i, j, atoms, com):
    s_vector = [0]*(3*atoms)
    e_ij=evector(com[i-1],com[j-1])
    s_vector[(3*(i-1)):(3*(i-1)+3)]=e_ij
    s_vector[(3*(j-1)):(3*(j-1)+3)]=-e_ij
    return s_vector

def angles_svectors(i, j, k, atoms, com):
    s_vector = [0]*(3*atoms)
    e_ij=evector(com[i-1],com[j-1])
    e_kj=evector(com[k-1],com[j-1])
    vec_ij=vector(com[i-1],com[j-1])
    vec_kj=vector(com[k-1],com[j-1])
    len_ij=length(com[i-1],com[j-1])
    len_kj=length(com[k-1],com[j-1])
    if np.allclose(np.cross(vec_ij,vec_kj),np.zeros(3)) == False:
        w = np.cross(vec_ij, vec_kj)
    elif np.allclose(np.cross(vec_ij, vec_kj),np.zeros(3)) == True and np.allclose(np.cross(vec_ij, [1,-1,1]),np.zeros(3)) == False and np.allclose(np.cross(vec_kj, [1,-1,1]),np.zeros(3)) == False:
        w = np.cross(vec_ij, [1,-1,1])
    elif np.allclose(np.cross(vec_ij, vec_kj),np.zeros(3)) == True and np.allclose(np.cross(vec_ij, [1,-1,1]),np.zeros(3)) == True and np.allclose(np.cross(vec_kj, [1,-1,1]),np.zeros(3)) == True:
        w = np.cross(vec_ij, [-1,1,1])
    w=w/np.sqrt(np.dot(w,w))
    s_vector[(3*(i-1)):(3*(i-1)+3)]=np.cross(e_ij,w)/length(com[i-1],com[j-1])
    s_vector[(3*(j-1)):(3*(j-1)+3)]=-np.cross(e_ij,w)/len_ij - np.cross(w,e_kj)/len_kj
    s_vector[(3*(k-1)):(3*(k-1)+3)]=np.cross(w,e_kj)/len_kj
    return s_vector    

def dihed_svectors(i, j, k, l, atoms, com):
    s_vector = [0]*(3*atoms)
    len_ij=length(com[i-1], com[j-1])
    len_lk=length(com[l-1], com[k-1])
    len_kj=length(com[k-1], com[j-1])
    e_ij=evector(com[i-1], com[j-1])        # u
    e_lk=evector(com[l-1], com[k-1])        # v
    e_kj=evector(com[k-1], com[j-1])        # w
    angle_ijk=np.arccos(np.dot(e_ij,e_kj))  # uw
    angle_jkl=np.arccos(np.dot(-e_lk,e_kj)) # vw
    s_vector[(3*(i-1)):(3*(i-1)+3)] = np.cross(e_ij,e_kj)/(len_ij*np.sin(angle_ijk)**2)
    s_vector[(3*(j-1)):(3*(j-1)+3)] = -np.cross(e_ij,e_kj)/(len_ij*np.sin(angle_ijk)**2) + ((np.cross(e_ij,e_kj)*np.cos(angle_ijk))/(len_kj*np.sin(angle_ijk)**2)-((np.cross(e_lk,e_kj)*np.cos(angle_jkl))/(len_kj*np.sin(angle_jkl)**2)))
    s_vector[(3*(k-1)):(3*(k-1)+3)] = np.cross(e_lk,e_kj)/(len_lk*np.sin(angle_jkl)**2) - ((np.cross(e_ij,e_kj)*np.cos(angle_ijk))/(len_kj*np.sin(angle_ijk)**2)-((np.cross(e_lk,e_kj)*np.cos(angle_jkl))/(len_kj*np.sin(angle_jkl)**2)))
    s_vector[(3*(l-1)):(3*(l-1)+3)] = -np.cross(e_lk,e_kj)/(len_lk*np.sin(angle_jkl)**2)
    return s_vector

def bond_b_tensor(i,j, atoms, com):
    b_tensor = np.zeros((3*atoms,3*atoms))
    e_ij = evector(com[i-1], com[j-1])
    len_ij = length(com[i-1], com[j-1])
    b_temp = np.outer(e_ij,e_ij)
    for a in range(3):
        for b in range(3):
            if a==b:
                krnckr = 1
            else:
                krnckr = 0
            b_temp[a][b] = (b_temp[a][b]-krnckr)/len_ij
    b_tensor[3*(i-1):(3*(i-1)+3),3*(i-1):(3*(i-1)+3)] = -1 * b_temp
    b_tensor[3*(i-1):(3*(i-1)+3),3*(j-1):(3*(j-1)+3)] =  1 * b_temp     
    b_tensor[3*(j-1):(3*(j-1)+3),3*(i-1):(3*(i-1)+3)] = -1 * b_temp
    b_tensor[3*(j-1):(3*(j-1)+3),3*(j-1):(3*(j-1)+3)] =  1 * b_temp
    return b_tensor

def angle_b_tensor(i,j,k, atoms, com):
    b_tensor = np.zeros((3*atoms,3*atoms))
    svector=angles_svectors(i,j,k,atoms,com)
    e_ij=evector(com[i-1],com[j-1])
    e_kj=evector(com[k-1],com[j-1])
    len_ij = length(com[i-1],com[j-1])
    len_kj = length(com[k-1],com[j-1])
    cosq_a = np.dot(e_ij,e_kj)
    sinq_a = np.sqrt(1-np.dot(e_ij,e_kj)**2)
    for a in range(3):
        for b in range(3):
            if a == b:
                krnckr = 1
            else:
                krnckr = 0
            first_term  = (e_ij[a]*e_kj[b] + e_ij[b]*e_kj[a] - 3*e_ij[a]*e_ij[b]*cosq_a + krnckr*cosq_a ) / (len_ij**2*sinq_a)
            second_term = (e_kj[a]*e_ij[b] + e_kj[b]*e_ij[a] - 3*e_kj[a]*e_kj[b]*cosq_a + krnckr*cosq_a ) / (len_kj**2*sinq_a)
            third_term  = (e_ij[a]*e_ij[b] + e_kj[b]*e_kj[a] - e_ij[a]*e_kj[b]*cosq_a - krnckr ) / (len_ij*len_kj*sinq_a)
            fourth_term = (e_kj[a]*e_kj[b] + e_ij[b]*e_ij[a] - e_kj[a]*e_ij[b]*cosq_a - krnckr ) / (len_ij*len_kj*sinq_a)
            
            #i-i,j,k terms
            b_tensor[3*(i-1)+a][3*(i-1)+b] = first_term - cosq_a/sinq_a * svector[(3*(i-1)+a)] * svector[3*(i-1)+b]
            b_tensor[3*(i-1)+a][3*(j-1)+b] = -first_term - third_term - cosq_a/sinq_a * svector[(3*(i-1)+a)] * svector[3*(j-1)+b]
            b_tensor[3*(i-1)+a][3*(k-1)+b] =  third_term - cosq_a/sinq_a * svector[(3*(i-1)+a)] * svector[3*(k-1)+b]
            
            #j-i,j,k terms
            b_tensor[3*(j-1)+a][3*(i-1)+b] = -first_term - fourth_term - cosq_a/sinq_a * svector[(3*(j-1)+a)] * svector[3*(i-1)+b]
            b_tensor[3*(j-1)+a][3*(j-1)+b] =  first_term + second_term - third_term - fourth_term - cosq_a/sinq_a * svector[(3*(j-1)+a)] * svector[3*(j-1)+b]
            b_tensor[3*(j-1)+a][3*(k-1)+b] = -second_term - third_term - cosq_a/sinq_a * svector[(3*(j-1)+a)] * svector[3*(k-1)+b]
            #k-i,j,k terms
            b_tensor[3*(k-1)+a][3*(i-1)+b] =  fourth_term - cosq_a/sinq_a * svector[(3*(k-1)+a)] * svector[3*(i-1)+b]
            b_tensor[3*(k-1)+a][3*(j-1)+b] = -second_term - fourth_term - cosq_a/sinq_a * svector[(3*(k-1)+a)] * svector[3*(j-1)+b]
            b_tensor[3*(k-1)+a][3*(k-1)+b] =  second_term - cosq_a/sinq_a * svector[(3*(k-1)+a)] * svector[3*(k-1)+b]
    return b_tensor

# i,i =  1 + 0 + 0 + 0
# i,j = -1 + 0 - 1 + 0
# i,k =  0 + 0 + 1 + 0

# j,i = -1 + 0 + 0 - 1
# j,j =  1 + 1 - 1 + 1
# j,k =  0 - 1 - 1 - 0

# k,i =  0 + 0 + 0 + 1
# k,j =  0 - 1 + 0 - 1
# k,k =  0 + 1 + 0 + 0

def dihed_b_tensor(i,j,k,l, atoms, com):
    b_tensor = np.zeros((3*atoms,3*atoms))
    len_ij=length(com[i-1], com[j-1])
    len_lk=length(com[l-1], com[k-1])
    len_kj=length(com[k-1], com[j-1])
    e_ij=evector(com[i-1], com[j-1])        # u
    e_lk=evector(com[l-1], com[k-1])        # v
    e_kj=evector(com[k-1], com[j-1])        # w
    cosq_u=np.dot(e_ij,e_kj)
    cosq_v=np.dot(-e_lk,e_kj)
    sinq_u=np.sqrt(1-np.dot(e_ij,e_kj)**2)
    sinq_v=np.sqrt(1-np.dot(e_lk,e_kj)**2)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if c == a or c == b:
                    continue
                else:
                    first_term   = ((np.cross(e_ij,e_kj)[a]*(e_kj[b]*cosq_u-e_ij[b])) + (np.cross(e_ij,e_kj)[b]*(e_kj[a]*cosq_u-e_ij[a]))) / (len_ij**2*sinq_u**4)
                    second_term  = ((np.cross(e_lk,e_kj)[a]*(e_kj[b]*cosq_v-e_lk[b])) + (np.cross(e_lk,e_kj)[b]*(e_kj[a]*cosq_v-e_lk[a]))) / (len_lk**2*sinq_v**4)
                    third_term   = ((np.cross(e_ij,e_kj)[a]*(e_kj[b] - 2*e_ij[b]*cosq_u + e_kj[b]*cosq_u**2)) + (np.cross(e_ij,e_kj)[b]*(e_kj[a] - 2*e_ij[a]*cosq_u + e_kj[a]*cosq_u**2))) / (2*len_ij*len_kj*sinq_u**4)
                    fourth_term  = ((np.cross(e_lk,e_kj)[a]*(e_kj[b] + 2*e_ij[b]*cosq_v + e_kj[b]*cosq_v**2)) + (np.cross(e_ij,e_lk)[b]*(e_kj[a] + 2*e_ij[a]*cosq_v + e_kj[a]*cosq_v**2))) / (2*len_lk*len_kj*sinq_v**4)
                    fifth_term   = ((np.cross(e_ij,e_kj)[a]*(e_ij[b] + e_ij[b]*cosq_u**2 - 3*e_kj[b]*cosq_u + e_kj[b]*cosq_u**3)) + (np.cross(e_ij,e_kj)[b]*(e_ij[a] + e_ij[a]*cosq_u**2 - 3*e_kj[a]*cosq_u + e_kj[a]*cosq_u**3))) / (2*len_kj**2*sinq_u**4)
                    sixth_term   = ((np.cross(e_lk,e_kj)[a]*(e_lk[b] + e_lk[b]*cosq_v**2 + 3*e_kj[b]*cosq_v - e_kj[b]*cosq_v**3)) + (np.cross(e_lk,e_kj)[b]*(e_lk[a] + e_lk[a]*cosq_v**2 + 3*e_kj[a]*cosq_v - e_kj[a]*cosq_v**3))) / (2*len_kj**2*sinq_u**4)
                    seventh_term = (b-a)*(-0.5)**(np.abs(b-a))*(e_kj[c]*cosq_u-e_ij[c]) / (len_ij*len_kj*sinq_u)
                    eighth_term  = (b-a)*(-0.5)**(np.abs(b-a))*(e_kj[c]*cosq_v-e_lk[c]) / (len_lk*len_kj*sinq_v)
                    
                    ## These take into account the presumed typo in the coefficient in front of
                    ## eighth term in Bakken&Helgaker 2002
                    #i-i,j,k,l terms
                    # b_tensor[3*(i-1)+a][3*(i-1)+b] =  first_term
                    # b_tensor[3*(i-1)+a][3*(j-1)+b] = -first_term + third_term + seventh_term
                    # b_tensor[3*(i-1)+a][3*(k-1)+b] = -third_term - seventh_term 
                    # b_tensor[3*(i-1)+a][3*(l-1)+b] = 0
                    # #j-i,j,k,l terms
                    # b_tensor[3*(j-1)+a][3*(i-1)+b] = -first_term + third_term + seventh_term
                    # b_tensor[3*(j-1)+a][3*(j-1)+b] =  first_term - 2*third_term - fifth_term + sixth_term - 2*seventh_term
                    # b_tensor[3*(j-1)+a][3*(k-1)+b] =  third_term + fourth_term + fifth_term - sixth_term + seventh_term + eighth_term
                    # b_tensor[3*(j-1)+a][3*(l-1)+b] = -fourth_term - eighth_term
                    # #k-i,j,k,l terms
                    # b_tensor[3*(k-1)+a][3*(i-1)+b] = -third_term - seventh_term
                    # b_tensor[3*(k-1)+a][3*(j-1)+b] =  third_term + fourth_term + fifth_term - sixth_term + seventh_term + eighth_term
                    # b_tensor[3*(k-1)+a][3*(k-1)+b] =  second_term - 2*fourth_term - fifth_term + sixth_term - 2*eighth_term
                    # b_tensor[3*(k-1)+a][3*(l-1)+b] = -second_term + fourth_term + eighth_term
                    # #l-i,j,k,l terms
                    # b_tensor[3*(l-1)+a][3*(i-1)+b] = 0
                    # b_tensor[3*(l-1)+a][3*(j-1)+b] = -fourth_term + eighth_term
                    # b_tensor[3*(l-1)+a][3*(k-1)+b] = -second_term + fourth_term - eighth_term
                    # b_tensor[3*(l-1)+a][3*(l-1)+b] =  second_term
                    
                    ## These are according to Bakken&Helgaker 2002 equations
                    ## (LHS are wrong, see correct form below...)

                    ##i-i,j,k,l terms
                    b_tensor[3*(i-1):(3*(i-1)+3),3*(i-1):(3*(i-1)+3)] =  first_term
                    b_tensor[3*(i-1):(3*(i-1)+3),3*(j-1):(3*(j-1)+3)] = -first_term + third_term + seventh_term
                    b_tensor[3*(i-1):(3*(i-1)+3),3*(k-1):(3*(k-1)+3)] = -third_term - seventh_term 
                    b_tensor[3*(i-1):(3*(i-1)+3),3*(l-1):(3*(l-1)+3)] = 0
                    #j-i,j,k,l terms
                    b_tensor[3*(j-1):(3*(j-1)+3),3*(i-1):(3*(i-1)+3)] = -first_term + third_term + seventh_term + eighth_term
                    b_tensor[3*(j-1):(3*(j-1)+3),3*(j-1):(3*(j-1)+3)] =  first_term - 2*third_term - fifth_term + sixth_term -2*seventh_term
                    b_tensor[3*(j-1):(3*(j-1)+3),3*(k-1):(3*(k-1)+3)] =  third_term + fourth_term + fifth_term - sixth_term + seventh_term + eighth_term
                    b_tensor[3*(j-1):(3*(j-1)+3),3*(l-1):(3*(l-1)+3)] = -fourth_term
                    #k-i,j,k,l terms
                    b_tensor[3*(k-1):(3*(k-1)+3),3*(i-1):(3*(i-1)+3)] = -third_term - seventh_term + eighth_term
                    b_tensor[3*(k-1):(3*(k-1)+3),3*(j-1):(3*(j-1)+3)] =  third_term + fourth_term - fifth_term - sixth_term + seventh_term + eighth_term
                    b_tensor[3*(k-1):(3*(k-1)+3),3*(k-1):(3*(k-1)+3)] =  second_term - 2*fourth_term - fifth_term + sixth_term
                    b_tensor[3*(k-1):(3*(k-1)+3),3*(l-1):(3*(l-1)+3)] = -second_term + fourth_term 
                    #l-i,j,k,l terms
                    b_tensor[3*(l-1):(3*(l-1)+3),3*(i-1):(3*(i-1)+3)] = 0
                    b_tensor[3*(l-1):(3*(l-1)+3),3*(j-1):(3*(j-1)+3)] = -fourth_term + eighth_term
                    b_tensor[3*(l-1):(3*(l-1)+3),3*(k-1):(3*(k-1)+3)] = -second_term + fourth_term - eighth_term
                    b_tensor[3*(l-1):(3*(l-1)+3),3*(l-1):(3*(l-1)+3)] =  second_term
                    
                    
                    
    return b_tensor



def get_bmat_and_cmat(internal_coordinates,      # supplied by a definition file currently
                      com):                      # center-of-mass-coordinates
    atoms = len(com)
    file = open(internal_coordinates)         
    bonds_array=angles_array=diheds_array=[]
    bonds_tensor=angles_tensor=diheds_tensor=[]
    i=j=k=0
    while True:
        user_input = file.readline().split(',')
        if user_input == ['']:
            print("Internal coordinates generated from Cartesian coordinates and connectivity information from",internal_coordinates)
            break
        if len(user_input) == 2:
            bond = bonds_svectors(int(user_input[0]), int(user_input[1]), atoms, com)
            bond_t = bond_b_tensor(int(user_input[0]), int(user_input[1]), atoms, com)
            if i == 0:
                bonds_array = bond
                bonds_tensor = bond_t
            else:
                bonds_array = np.vstack((bonds_array, bond))
                bonds_tensor = np.dstack((bonds_tensor, bond_t))
            i+=1
        if len(user_input) == 3:
            angle = angles_svectors(int(user_input[0]), int(user_input[1]), int(user_input[2]), atoms, com)
            angle_t = angle_b_tensor(int(user_input[0]), int(user_input[1]), int(user_input[2]), atoms, com)
            if j == 0:
                angles_array = angle
                angles_tensor = angle_t
            else:
                angles_array = np.vstack((angles_array, angle))
                angles_tensor = np.dstack((angles_tensor, angle_t))
            j+=1
        if len(user_input) == 4:
            dihed = dihed_svectors(int(user_input[0]), int(user_input[1]), int(user_input[2]), int(user_input[3]), atoms, com)
            dihed_t = dihed_b_tensor(int(user_input[0]), int(user_input[1]), int(user_input[2]), int(user_input[3]), atoms, com)        
            if k == 0:
                diheds_array = dihed
                diheds_tensor = dihed_t
            else:
                diheds_array = np.vstack((diheds_array, dihed))
                diheds_tensor = np.dstack((diheds_tensor, dihed_t))
            k+=1
    
    if len(diheds_array) == 0 and len(angles_array) == 0 and len(bonds_array) == 6:
        bmat=np.asarray(bonds_array)
        bmat=np.reshape(bmat, (1,len(bmat)))
        cmat=np.asarray(bonds_tensor)
        cmat=np.reshape(cmat, (len(cmat),len(cmat),1))
    elif len(diheds_array) == 0 and len(angles_array) == 0 and len(bonds_array) > 6:
        bmat=bonds_array
        cmat=bonds_tensor
    elif len(diheds_array) == 0 and len(angles_array) != 0:
        bmat=np.vstack((bonds_array, angles_array))
        cmat=np.dstack((bonds_tensor, angles_tensor))
    else:
        bmat=np.vstack((bonds_array, angles_array, diheds_array))
        cmat=np.dstack((bonds_tensor, angles_tensor, diheds_tensor))
    cmat=cmat.T
    return bmat,cmat


# def gf_freqs(cart_hess, mass_matrix, bmat, rot_sym):
#     u=np.linalg.inv(mass_matrix)
#     GF_G = np.matmul(np.matmul(bmat,u),bmat.T)
    
#     ## Remove redundancies from the G matrix
#     gval,gvec = np.linalg.eig(GF_G)
#     idx=gval.argsort()
#     idx=np.flip(idx)
#     gval=gval[idx]
#     gvec=gvec[:,idx]
#     U=[]
#     atoms=int(len(mass_matrix))
#     if rot_sym != 1:
#         for i in range(atoms-6):    # Non-linear molecule
#             U.append(gvec[i])
#     else:
#         for i in range(atoms-5):    # Linear molecule
#             U.append(gvec[i])
#     U=np.asarray(U)
#     U=np.real(U.T)
#     bmat = np.matmul(U.T,bmat)
#     GF_G = np.matmul(np.matmul(bmat,u),bmat.T)
    
#     A = np.matmul(np.matmul(u,bmat.T),np.linalg.inv(GF_G))
#     f = np.matmul(np.matmul(A.T,cart_hess),A)
#     fval,fvec=np.linalg.eig(np.matmul(GF_G,f))
#     idx=fval.argsort()
#     fval = fval[idx]
#     fvec = fvec[:,idx]
#     return fval,fvec

def gf_freqs(cart_hess, mass_matrix, bmat, rot_sym):
    u = np.linalg.inv(mass_matrix)
    G = bmat.dot(u).dot(bmat.T)
    G,G_inv = tools.get_G_matrix_and_G_inv_matrix(G)
    A = u.dot(bmat.T).dot(G_inv)
    f = A.T.dot(cart_hess).dot(A)
    P = G.dot(G_inv)
    f=P.dot(f).dot(P)
    feigval,feigvec=np.linalg.eig(G.dot(f))
    idx = np.flip(np.abs(feigval).argsort())
    feigval=feigval[idx]
    feigvec=feigvec[idx]
    fval=[]
    fvec=[]
    if rot_sym == 1:
        for i in range(len(mass_matrix)-5):
            fval.append(feigval[i])
            fvec.append(feigvec[i])
    else:
        for i in range(len(mass_matrix)-6):
            fval.append(feigval[i])
            fvec.append(feigvec[i])
    fval=np.asarray(fval)
    fvec=np.asarray(fval)
    fval=fval[fval.argsort()]
    fvec=fvec[fval.argsort()]
    return (fval, fvec)


def gf_proj_freqs(cart_hess, cart_grad, mass_matrix, bmat, cmat, rot_sym):
    u=np.linalg.inv(mass_matrix)
    G = bmat.dot(u).dot(bmat.T)
    G,G_inv = tools.get_G_matrix_and_G_inv_matrix(G)
    A = u.dot(bmat.T).dot(G_inv)
    f = A.T.dot(cart_hess).dot(A)
    g = A.T.dot(cart_grad)
    for a in range(len(g)):
        f = f - g[a]*np.matmul(np.matmul(A.T,cmat[a]),A)
    P=G.dot(G_inv)
    f=P.dot(f).dot(P)
    g=P.dot(g)
    
    """
    Construct a projector for the gradient
    p = gg.T / (g.T BuB.T g)
    """
    p = (g.dot(g.T))/(g.T.dot(G).dot(g))
    I=np.eye(len(p))
    pre = I-(p.dot(G))
    post = I-(G.dot(p))    
    f_proj = pre.dot(f).dot(post)
    """
    diagonalization of the Wilson GF matrix gives vibrations orthogonal
    to the internal gradient
    GF = GF_G * f_proj
    """
    feigval,feigvec=np.linalg.eig(G.dot(f_proj))
    idx = np.flip(np.abs(feigval).argsort())
    feigval=feigval[idx]
    feigvec=feigvec[idx]
    fval=[]
    fvec=[]
    if rot_sym == 1:
        for i in range(len(mass_matrix)-5):
            fval.append(feigval[i])
            fvec.append(feigvec[i])
    else:
        for i in range(len(mass_matrix)-6):
            fval.append(feigval[i])
            fvec.append(feigvec[i])
    fval=np.asarray(fval)
    fvec=np.asarray(fval)
    fval=fval[fval.argsort()]
    fvec=fvec[fval.argsort()]
    return (fval, fvec)

"""
This section deals with redundant internal coordinates.

Redundant internals are currently not supported, because the cmat slices of angles would depend
on the corresponding B-matrix elements, which for redundant internals are obtained 
by the SVD of B_red*B_red.T, that yields the non-redundant set of delocalized internals.

For stationary points, where the Hessian in principle does not depend on cmat,
the redundant internals are ok and the below code can be used. If so, then these lines
below need to be commented out:

#
for a in range(len(g)):
    f = f - g[a]*np.matmul(np.matmul(A.T,cmat[a]),A)
#


# Generation of non-redundant B-matrix from redundant internals
gmat = np.matmul(bmat,bmat.T)
gval,gvec=np.linalg.eig(gmat)
index=gval.argsort()
index=np.flip(index)
gval = gval[index]
gvec = gvec[:,index]
U=[]
for i in range(3*num_atoms-6):
        U.append(gvec[i])
U=np.asarray(U)
U=np.real(U.T)
bmat=np.matmul(U.T,bmat)
"""


## These are according to Bakken&Helgaker 2002 equations
## (LHS are wrong, see correct form below...)

##i-i,j,k,l terms
#b_tensor[3*(i-1):(3*(i-1)+3),3*(i-1):(3*(i-1)+3)] =  first_term
#b_tensor[3*(i-1):(3*(i-1)+3),3*(j-1):(3*(j-1)+3)] = -first_term + third_term + seventh_term
#b_tensor[3*(i-1):(3*(i-1)+3),3*(k-1):(3*(k-1)+3)] = -third_term - seventh_term 
#b_tensor[3*(i-1):(3*(i-1)+3),3*(l-1):(3*(l-1)+3)] = 0
#j-i,j,k,l terms
#b_tensor[3*(j-1):(3*(j-1)+3),3*(i-1):(3*(i-1)+3)] = -first_term + third_term + seventh_term + eighth_term
#b_tensor[3*(j-1):(3*(j-1)+3),3*(j-1):(3*(j-1)+3)] =  first_term - 2*third_term - fifth_term + sixth_term -2*seventh_term
#b_tensor[3*(j-1):(3*(j-1)+3),3*(k-1):(3*(k-1)+3)] =  third_term + fourth_term + fifth_term - sixth_term + seventh_term + eighth_term
#b_tensor[3*(j-1):(3*(j-1)+3),3*(l-1):(3*(l-1)+3)] = -fourth_term
#k-i,j,k,l terms
#b_tensor[3*(k-1):(3*(k-1)+3),3*(i-1):(3*(i-1)+3)] = -third_term - seventh_term + eighth_term
#b_tensor[3*(k-1):(3*(k-1)+3),3*(j-1):(3*(j-1)+3)] =  third_term + fourth_term - fifth_term - sixth_term + seventh_term + eighth_term
#b_tensor[3*(k-1):(3*(k-1)+3),3*(k-1):(3*(k-1)+3)] =  second_term - 2*fourth_term - fifth_term + sixth_term
#b_tensor[3*(k-1):(3*(k-1)+3),3*(l-1):(3*(l-1)+3)] = -second_term + fourth_term 
#l-i,j,k,l terms
#b_tensor[3*(l-1):(3*(l-1)+3),3*(i-1):(3*(i-1)+3)] = 0
#b_tensor[3*(l-1):(3*(l-1)+3),3*(j-1):(3*(j-1)+3)] = -fourth_term + eighth_term
#b_tensor[3*(l-1):(3*(l-1)+3),3*(k-1):(3*(k-1)+3)] = -second_term + fourth_term - eighth_term
#b_tensor[3*(l-1):(3*(l-1)+3),3*(l-1):(3*(l-1)+3)] =  second_term

# m o p n 
# i j k l

# 1. amo*bmo ( m - o x m - o ) mo mo (0) (0)

# 2. anp*bnp ( n - p x n - p ) np np (0) (0)

# 3. amo*bop + apo*bom ( m - o x o - p ) + ( p - o x o - m ) moop poom (1) (1)

# 4. anp*bpo + apo*bnp ( n - p x p - o ) + ( p - o x n - p ) nppo ponp (1) (1)

# 5. aop*bpo ( o - p x p - o ) op po (1) (1)

# 6. aop*bop ( o - p x o - p ) op op (-1) (-1)

# 7. (1-ab)*(amo*bop + apo*bom)  ( m - o x o - p ) + ( p - o x o - m ) moop poom (1) (1)

# 8. (1-ab)*(ano*bop + apo*bom)  ( n - o x o - p ) + ( p - o x o - m ) noop poom (1) (1)

# 8th is actully likely ( n - p x p - o ) + ( p - o x n - p ) (1) (1)



## Bakken & Helgaker 2002
# i,i =  1 + 0 + 0 + 0 + 0 + 0 + 0 + 0
# i,j = -1 + 0 + 1 + 0 + 0 + 0 + 1 + 0
# i,k =  0 + 0 - 1 + 0 + 0 + 0 - 1 + 0
# i,l =  0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 

# j,i = -1 + 0 + 1 + 0 + 0 + 0 + 1 + 1
# j,j =  1 + 0 - 2 + 0 - 1 + 1 + 0 + 0
# j,k =  0 + 0 + 1 + 1 + 1 - 1 + 1 + 1 
# j,l =  0 + 0 + 0 - 1 + 0 + 0 + 0 + 0

# k,i =  0 + 0 - 1 + 0 + 0 + 0 - 1 + 1
# k,j =  0 + 0 + 1 + 1 + 1 - 1 + 1 + 1 
# k,k =  0 + 1 + 0 - 2 - 1 + 1 + 0 + 0
# k,l =  0 - 1 + 0 + 1 + 0 + 0 + 0 + 0 

# l,i =  0 + 0 + 0 + 0 + 0 + 0 + 0 + 0
# l,j =  0 + 0 + 0 - 1 + 0 + 0 + 0 + 1
# l,k =  0 - 1 + 0 + 1 + 0 + 0 + 0 - 1
# l,l =  0 + 1 + 0 + 0 + 0 + 0 + 0 + 0


## Corrected version
# i,i =  1 + 0 + 0 + 0 + 0 + 0 + 0 + 0
# i,j = -1 + 0 + 1 + 0 + 0 + 0 + 1 + 0
# i,k =  0 + 0 - 1 + 0 + 0 + 0 - 1 + 0
# i,l =  0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 

# j,i = -1 + 0 + 1 + 0 + 0 + 0 + 1 + 0
# j,j =  1 + 0 - 2 + 0 - 1 + 1 - 2 + 0
# j,k =  0 + 0 + 1 + 1 + 1 - 1 + 1 + 1 
# j,l =  0 + 0 + 0 - 1 + 0 + 0 + 0 - 1

# k,i =  0 + 0 - 1 + 0 + 0 + 0 - 1 + 0
# k,j =  0 + 0 + 1 + 1 + 1 - 1 + 1 + 1 
# k,k =  0 + 1 + 0 - 2 - 1 + 1 + 0 - 2
# k,l =  0 - 1 + 0 + 1 + 0 + 0 + 0 + 0 

# l,i =  0 + 0 + 0 + 0 + 0 + 0 + 0 + 0
# l,j =  0 + 0 + 0 - 1 + 0 + 0 + 0 + 1
# l,k =  0 - 1 + 0 + 1 + 0 + 0 + 0 - 1
# l,l =  0 + 1 + 0 + 0 + 0 + 0 + 0 + 0
