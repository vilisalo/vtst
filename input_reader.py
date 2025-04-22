import numpy as np



##### THESE ARE ORCA RELATED FUNCTIONS #####
def hessian_reader(hess_file):
    coord=[]
    with open(hess_file, 'r') as inp:
        for line in inp:
            if not "$hessian" in line: # Flag for the Hessian group
                continue
            else:
                for data in inp:
                    ##--- Important Parameters
                    hess_size = int(data.strip())
                    n5_sets = hess_size//5 # Number of 5 columns sets. 
                    rest = hess_size%5
                    if rest == 0:
                        n5_sets = n5_sets-1 # If the size is divisible by 5, then the sets would be larger.
                        rest = 5
                    ## Create the HESSIAN array
                    hess_data = np.zeros((hess_size,hess_size+1))
                    ## Write the number of the output line in the first element of the hess_data matrix
                    for n_line in range(hess_size):
                        hess_data[n_line][0] = n_line+1
                    next(inp) # Jump the second line.
                    break
                #for data in inp: # Jump the second line (improve this)
                #    break
                count = 0
                while count <= n5_sets:
                    count2 = 0
                    ## First get the data from all sets of 5 columns except the last one if rest
                    # is not zero.
                    if count != n5_sets:
                        for data in inp:
                            columns = [ int(num) for num in range(count*5,(count)*5+5) ]
                            data = data.split()
                            n_line = int(data[0])
                            if  count2 >= hess_size:
                                break
                            for i in range(5):
                                j = columns[i] + 1
                                hess_data[n_line][j] = float(data[i+1])
                            count2 += 1
                    ## This is the condition for the last sets of column that will not be zero
                    # if the size of the HESS matrix is not divisible by 5.
                    else:
                        for data in inp:
                            columns = [ int(num) for num in range(count*5,(count)*5+rest) ]
                            if data.strip() == '':
                                break
                            data = data.split()
                            n_line = int(data[0])
                            if  count2 >= hess_size:
                                break
                            for i in range(rest):
                                j = columns[i] + 1
                                hess_data[n_line][j] = float(data[i+1])
                            count2 += 1
                    count += 1                  
                    
                    
            for line in inp:
                if not "$atoms" in line:
                    continue
                num_atoms = inp.readline()
                for line in range(int(num_atoms)):
                    line1=inp.readline()
                    atom,mass,x,y,z = line1.split()
                    coord.append([atom, float(mass), float(x), float(y), float(z)])
    coord=np.asarray(coord)
    hess_data = np.delete(hess_data, 0, 1)
    num_atoms=int(num_atoms)
    
    M_mat = np.zeros( (3*num_atoms,3*num_atoms), dtype = float )
    W_mat = np.zeros( (3*num_atoms,3*num_atoms), dtype = float )
    mass_triples = [ coord[i,1] for i in range(num_atoms) for j in range(3)]
    mass_triples = np.asarray(mass_triples)
    mass_triples = mass_triples.astype(float)  
    for i in range(len(mass_triples)):
        M_mat[i,i] = mass_triples[i]
    
    return (num_atoms, coord, hess_data, M_mat)
    ###############################################################################
    
    
def grad_reader(grad_file):
    grad=[]
    ############### SECTION BELOW READS THE GRADIENT FILE FROM ORCA ###############
    with open(grad_file, 'r') as inp:
        for line in inp:
            if not "# Number of atoms" in line:
                continue
            inp.readline()
            num_atoms = inp.readline()
    with open(grad_file, 'r') as inp:
        for line in inp:
            if not "# The current gradient" in line: # Flag for the start of Gradient block
                continue
            inp.readline()
            for i in range(3*int(num_atoms)):
                N=inp.readline()
                N=N.lstrip()
                N=N.rstrip()
                N=float(N)
                grad.append(N)
    grad=np.asarray(grad)
    grad=np.reshape(grad, (3*int(num_atoms),1))
    return grad


def b_matrix_reader(opt_file):
    with open(opt_file, 'r') as inp:
        for line in inp:
            if not "$bmatrix" in line: # Flag for the b-matrix group
                continue
            else:
                for data in inp:
                    ##--- Important Parameters                 
                    bmat_size = data.split()
                    bmat_size = np.asarray(bmat_size)
                    bmat_size = bmat_size.astype(int)
                    n6_sets = bmat_size[1]//6 # Number of 6 columns sets. 
                    rest = bmat_size[1]%6
                    if rest == 0:
                        n6_sets = n6_sets-1 # If the size is divisible by 6, then the sets would be larger.
                        rest = 6
                    ## Create the BMATRIX array
                    bmat_data = np.zeros((bmat_size[0],bmat_size[1]+1))
                    ## Write the number of the output line in the first element of the bmat_data matrix
                    for n_line in range(bmat_size[0]):
                        bmat_data[n_line][0] = n_line+1
                    next(inp) # Jump the second line.
                    break

                #for data in inp: # Jump the second line (improve this)
                #    break
                count = 0
                while count <= n6_sets:
                    count2 = 0
                    ## First get the data from all sets of 6 columns except the last one if rest
                    # is not zero.
                    if count != n6_sets:
                        for data in inp:
                            columns = [ int(num) for num in range(count*6,(count)*6+6) ]
                            data = data.split()
                            n_line = int(data[0])
                            if  count2 >= bmat_size[0]:
                                break
                            for i in range(6):
                                j = columns[i] + 1
                                bmat_data[n_line][j] = float(data[i+1])
                            count2 += 1
                    ## This is the condition for the last sets of column that will not be zero
                    # if the size of the BMAT matrix is not divisible by 6.
                    else:
                        for data in inp:
                            columns = [ int(num) for num in range(count*6,(count)*6+rest) ]
                            if data.strip() == '':
                                break
                            data = data.split()
                            n_line = int(data[0])
                            if  count2 >= bmat_size[0]:     # CHECK THIS
                                break
                            for i in range(rest):
                                j = columns[i] + 1
                                bmat_data[n_line][j] = float(data[i+1])
                            count2 += 1
                    count += 1                  
    
    bmat_data = np.delete(bmat_data, 0, 1)
    return (bmat_data)

####################################################################

###### THESE ARE GAUSSIAN RELATED FUNCTIONS #######

def gaussian_parser(gaussian_fchk_file):
    with open(gaussian_fchk_file, 'r') as file:
        for line in file:
            
            if "Number of atoms" in line:
                num_atoms = int(" ".join(line.split()).split()[-1])
                
            if "Multiplicity" in line:
                mult = int(" ".join(line.split()).split()[-1])
                
            #### Here you should read the atomic symbols !!
            
            if "Current cartesian coordinates" in line:
                coord = []
                if (3*num_atoms)/5 >= 1:
                    if (3*num_atoms) % 5 != 0:
                        num_of_lines = int(3*num_atoms / 5 + 1)
                    else:
                        num_of_lines = int(3*num_atoms / 5)
                else:
                    num_of_lines = 1    
                for i in range(0,num_of_lines,1):
                    coord.extend(next(file).split())
                coord = np.asarray(coord, dtype=float)
                coord = np.reshape(coord, (3,num_atoms))
                
            if "Real atomic weights" in line:
                masses = []
                if num_atoms/5 >= 1:
                    if num_atoms % 5 != 0:
                        num_of_lines = int(num_atoms / 5 + 1)
                    else:
                        num_of_lines = int(num_atoms / 5)
                else:
                    num_of_lines = 1
                for i in range(0,num_of_lines,1):
                    masses.extend(next(file).split())
                masses = np.asarray(masses, dtype=float)
                mass_matrix = np.zeros( (3*num_atoms,3*num_atoms), dtype = float )
                mass_triples = [ masses[i] for i in range(num_atoms) for j in range(3)]
                mass_triples = np.asarray(mass_triples)
                mass_triples = mass_triples.astype(float)  
                for i in range(len(mass_triples)):
                    mass_matrix[i,i] = mass_triples[i]
                
            #### Here you should read the internal coordinates !!
            
            if "Cartesian Gradient" in line:
                gradient_cart = []
                if (3*num_atoms)/5 >= 1:
                    if (3*num_atoms) % 5 != 0:
                        num_of_lines = int(3*num_atoms / 5 + 1)
                    else:
                        num_of_lines = int(3*num_atoms / 5)
                else:
                    num_of_lines = 1    
                for i in range(0,num_of_lines,1):
                    gradient_cart.extend(next(file).split())
                gradient_cart = np.asarray(gradient_cart, dtype=float)
                
            if "Cartesian Force Constants" in line:
                lower_triangle = []
                hessian_cart = np.zeros((3*num_atoms,3*num_atoms))
                triangular_dim = triangular_number(3*num_atoms)
                if triangular_dim/5 >= 1:
                    if triangular_dim % 5 != 0:
                        num_of_lines = int(triangular_dim / 5 + 1)
                    else:
                        num_of_lines = int(triangular_dim / 5)
                else:
                    num_of_lines = 1
                for i in range(0,num_of_lines,1):
                    lower_triangle.extend(next(file).split())
                
                hessian_cart = triangle_to_square(lower_triangle)
                for i in range(3*num_atoms):
                    for j in range(3*num_atoms):
                        hessian_cart[i][j] = hessian_cart[j][i]
                
    file.close()
    return num_atoms, mult, coord, masses, gradient_cart, hessian_cart, mass_matrix
    #### Still requires internal coordinates and atom symbols 




def triangular_number(n):
    i, t = 1, 0
    while i <= n:
        t += i
        i += 1
    return t

def triangle_to_square(triangle):
    n = int(np.sqrt(len(triangle)*2))
    mask = np.tri(n,dtype=bool, k=0)
    out = np.zeros((n,n),dtype=float)
    out[mask] = triangle
    return out
    
 
