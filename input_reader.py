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

class gaussian_parser:
    def __init__(self, gaussian_fchk_file):
        with open(gaussian_fchk_file, 'r') as file:
            for line in file:
                
                if "Number of atoms" in line:
                    self.num_atoms = int(" ".join(line.split()).split()[-1])
                    
                if "Multiplicity" in line:
                    self.mult = int(" ".join(line.split()).split()[-1])
                    
                if "Atomic numbers" in line:
                    atoms = []
                    if self.num_atoms/6 >= 1:
                        if self.num_atoms % 6 != 0:
                            num_of_lines = int(self.num_atoms / 6 + 1)
                        else:
                            num_of_lines = int(self.num_atoms / 6)
                    else:
                        num_of_lines = 1
                    for i in range(0,num_of_lines,1):
                        atoms.extend(next(file).split())
                    atoms = np.asarray(atoms, dtype=str)
                    self.atoms = atoms
                
                if "Current cartesian coordinates" in line:
                    coord = []
                    if (3*self.num_atoms)/5 >= 1:
                        if (3*self.num_atoms) % 5 != 0:
                            num_of_lines = int(3*self.num_atoms / 5 + 1)
                        else:
                            num_of_lines = int(3*self.num_atoms / 5)
                    else:
                        num_of_lines = 1    
                    for i in range(0,num_of_lines,1):
                        coord.extend(next(file).split())
                    coord = np.asarray(coord, dtype=float)
                    self.coord = np.reshape(coord, (self.num_atoms,3))
                    
                if "Real atomic weights" in line:
                    masses = []
                    if self.num_atoms/5 >= 1:
                        if self.num_atoms % 5 != 0:
                            num_of_lines = int(self.num_atoms / 5 + 1)
                        else:
                            num_of_lines = int(self.num_atoms / 5)
                    else:
                        num_of_lines = 1
                    for i in range(0,num_of_lines,1):
                        masses.extend(next(file).split())
                    self.masses = np.asarray(masses, dtype=float)
                    mass_matrix = np.zeros( (3*self.num_atoms,3*self.num_atoms), dtype = float )
                    mass_triples = [ masses[i] for i in range(self.num_atoms) for j in range(3)]
                    mass_triples = np.asarray(mass_triples)
                    mass_triples = mass_triples.astype(float)  
                    for i in range(len(mass_triples)):
                        mass_matrix[i,i] = mass_triples[i]
                    self.mass_matrix = mass_matrix
                    
                #### Here you should read the internal coordinates !!
                if "Redundant internal coordinate indices" in line:
                    redundant_coordinates= []
                    num_of_indices = int(" ".join(line.split()).split()[-1])
                    if num_of_indices / 6 >= 1:
                        if num_of_indices % 6 != 0:
                            num_of_lines = int(num_of_indices / 6 + 1)
                        else:
                            num_of_lines = int(num_of_indices / 6)
                    else:
                        num_of_lines = 1
                    for i in range(0,num_of_lines,1):
                        redundant_coordinates.extend(next(file).split())
                    redundant_coordinates = np.asarray(redundant_coordinates, dtype=int)
                    redundant_coordinates = np.reshape(redundant_coordinates, (int(len(redundant_coordinates)/4), 4))                              
                    self.int = redundant_coordinates
                    with open("internal_coordinates.txt", "w") as f:
                        for i in self.int:
                            if i[2] == 0:
                                obj = ",".join(map(str,i[:2]))
                                f.write(obj+"\n")
                            else:
                                if i[3] == 0:
                                    obj = ",".join(map(str,i[:3]))
                                    f.write(obj+"\n")
                                else:
                                    obj = ",".join(map(str,i[:4]))
                                    f.write(obj+"\n")
                        f.close()
                                
                    
                if "Cartesian Gradient" in line:
                    gradient_cart = []
                    if (3*self.num_atoms)/5 >= 1:
                        if (3*self.num_atoms) % 5 != 0:
                            num_of_lines = int(3*self.num_atoms / 5 + 1)
                        else:
                            num_of_lines = int(3*self.num_atoms / 5)
                    else:
                        num_of_lines = 1    
                    for i in range(0,num_of_lines,1):
                        gradient_cart.extend(next(file).split())
                    gradient_cart = np.reshape(gradient_cart, (3*self.num_atoms,1))
                    self.gradient_cart = np.asarray(gradient_cart, dtype=float)
                    
                if "Cartesian Force Constants" in line:
                    lower_triangle = []
                    hessian_cart = np.zeros((3*self.num_atoms,3*self.num_atoms))
                    triangular_dim = triangular_number(3*self.num_atoms)
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
                    for i in range(3*self.num_atoms):
                        for j in range(3*self.num_atoms):
                            hessian_cart[i][j] = hessian_cart[j][i]
                    self.hessian_cart = hessian_cart
                    
        file.close()



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
    
 
