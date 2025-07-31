import warnings
import numpy as np
import gf_method
pm_to_bohr = 0.0188972599


def get_G_matrix_and_G_inv_matrix(matrix, tolerance=1e-7):
    U,S,VT = np.linalg.svd(matrix)
    S_inv=[]
    for i in range(len(S)):
        if S[i] < tolerance:
            S[i] = 0
            S_inv.append(0)
        else:
            S_inv.append(1/(S[i]))
    S_inv=np.asarray(S_inv, dtype=float)
    S=np.diag(S)
    S_inv=np.diag(S_inv)
    G = U.dot(S).dot(VT)
    G_inv = VT.T.dot(S_inv).dot(U.T)
    return G,G_inv
    
def get_center_mass(coord, masses):
    cbye = [np.dot(masses[i], coord[i]) for i in range(len(coord))]
    r = np.sum(cbye, axis=0)
    r = r / np.sum(masses)
    return r


def get_inertia_tensor(coord, masses):

    # Build inertia tensor
    inertia_matrix = np.zeros((3, 3))
    for m, c in zip(masses, coord):
        inertia_matrix += m * (np.identity(3) * np.dot(c, c) - np.outer(c, c))

    total_inertia = 0
    for idx, atom in enumerate(coord):
        total_inertia += masses[idx] * np.dot(atom, atom)

    if abs(total_inertia) > 0:
        inertia_matrix /= total_inertia
    return inertia_matrix

def print_coordinates(atoms, coord):
    atoms=np.reshape(atoms, (len(atoms),1))
    xyz = np.concatenate((atoms,coord),axis=1)
    for i in xyz:
        for j in i:
            print(j, end="\t")
        print()

def get_perpendicular(vector, tol=1e-8):
    index = np.argmin(np.abs(vector))
    p_vector = np.identity(3)[index]
    pp_vector = np.cross(vector, p_vector)
    pp_vector = pp_vector / np.linalg.norm(pp_vector)

    assert np.dot(pp_vector, vector) < tol  # check perpendicular
    assert abs(np.linalg.norm(pp_vector) - 1) < tol  # check normalized

    return pp_vector


def get_degeneracy(eigenvalues, tolerance=0.1):

    for ev1 in eigenvalues:
        single_deg = 0
        for ev2 in eigenvalues:
            if abs(ev1 - ev2) < tolerance:
                single_deg += 1
        if single_deg > 1:
            return single_deg
    return 1


def get_non_degenerated(eigenvalues, tolerance=0.1):

    for i, ev1 in enumerate(eigenvalues):
        single_deg = 0
        index = 0
        for ev2 in eigenvalues:
            if not abs(ev1 - ev2) < tolerance:
                single_deg += 1
                index = i
        if single_deg == 2:
            return index

    raise Exception('Non degenerate not found')


def magic_formula(n):
    return np.sqrt(n*2**(3-n))


def rotation_matrix(axis, angle):
    """
    rotation matrix

    :param axis: normalized axis
    :param angle: angle in radians
    :return:
    """

    norm = np.linalg.norm(axis)
    assert norm > 1e-8
    axis = np.array(axis) / norm  # normalize axis

    cos_term = 1 - np.cos(angle)
    rot_matrix = [[axis[0]**2*cos_term + np.cos(angle),              axis[0]*axis[1]*cos_term - axis[2]*np.sin(angle), axis[0]*axis[2]*cos_term + axis[1]*np.sin(angle)],
                  [axis[1]*axis[0]*cos_term + axis[2]*np.sin(angle), axis[1]**2*cos_term + np.cos(angle),              axis[1]*axis[2]*cos_term - axis[0]*np.sin(angle)],
                  [axis[2]*axis[0]*cos_term - axis[1]*np.sin(angle), axis[1]*axis[2]*cos_term + axis[0]*np.sin(angle), axis[2]**2*cos_term + np.cos(angle)]]

    return np.array(rot_matrix)


def get_symmetry_number(pointgroup):
    if pointgroup in ['C1', 'Cs', 'Cinfv']:
        return 1
    if pointgroup in ['C2', 'C2v', 'Dinfh']:
        return 2
    if pointgroup in ['C3', 'C3v', 'C3h']:
        return 3
    if pointgroup in ['D2h']:
        return 4
    if pointgroup in ['D3h', 'D3d']:
        return 6
    if pointgroup in ['D5h']:
        return 10
    if pointgroup in ['Td']:
        return 12
    if pointgroup in ['Oh']:
        return 24
    else:
        return print("The ",pointgroup," is not currently supported.")
    
def get_rot_const_from_I(Ieigval):
    c = 29979245800 # cm/s
    bohr_to_meter = 5.29177249E-11    # m/bohr
    h = 6.62607015E-34 # Js
    NA = 6.0221408E23 # 1/mol
    Ieigval = Ieigval[Ieigval.argsort()]
    Ieigval = Ieigval*bohr_to_meter**2
    B = h/(8*np.pi**2*c*Ieigval)*1000*NA
    return B


class get_redundant_internals:
    def __init__(self,coord,atoms):

        #### Bonds
        if len(atoms) >= 2:
            bonds=[]
            cov_rad=[]
            for i in range(len(atoms)):
                cov_rad.append(get_covalent_radius(atoms[i]))
            cov_rad = np.asarray(cov_rad, dtype=float)
            
            for i in range(len(coord)):
                for j in range(len(coord)):
                    if i != j and i < j:
                        if gf_method.length(coord[i],coord[j]) <= 1.3*cov_rad[i]+cov_rad[j]:
                            bonds.append([i+1,j+1]) 
            bonds=np.asarray(bonds, dtype=int)
            #### NEW ADDITION, NOT TESTED PROPERLY
            intramolecular_bonds = get_connectivity(bonds, coord)
            if len(intramolecular_bonds) != 0:
                bonds = np.append(bonds, intramolecular_bonds, axis=0)

        #### Angles
        if len(bonds) >= 2:
            angles=[]
            for i in range(len(bonds)):
                for j in range(len(bonds)):
                    if i != j and i < j:
                        if any(item in bonds[i] for item in bonds[j]) == True:
                            order = np.nonzero(np.in1d(bonds[i],bonds[j]))[0][0]
                            if order == 0:
                                concatenated = np.concatenate((np.flip(bonds[i]),bonds[j]))
                                idx=np.unique(concatenated, return_index=True)[1]
                                angles.append([concatenated[i] for i in sorted(idx)])
                            if order == 1:
                                concatenated = np.concatenate((bonds[i],bonds[j]))
                                idx=np.unique(concatenated, return_index=True)[1]
                                angles.append([concatenated[i] for i in sorted(idx)])
            angles=np.asarray(angles, dtype=int)
            
        #### Dihedrals and improper torsions
        if len(angles) >= 2:
            dihedrals=[]
            for i in range(len(angles)):
                for j in range(len(angles)):
                    if i != j and i < j:
                        if any(item in angles[i] for item in angles[j]) == True:
                                if len(np.nonzero(np.in1d(angles[i],angles[j]))[0]) == 2:
                                    
                                    ### This condition applies to proper dihedrals
                                    if any(angles[i] -angles[j] == 0) == False:
                                        ### Permute atoms such that they match with linear bonding pattern
                                        concatenated=np.concatenate((angles[i],angles[j]))
                                        idx=np.unique(concatenated, return_index=True)[1]
                                        trial_dihedral = np.array(([concatenated[i] for i in sorted(idx)]), dtype=int)
                                        trial_dihedral = np.array(([trial_dihedral[0],trial_dihedral[1]], [trial_dihedral[1],trial_dihedral[2]], [trial_dihedral[2],trial_dihedral[3]]), dtype=int)                                        
                                        if isSubset(trial_dihedral, bonds) == True:
                                            dihedrals.append([concatenated[i] for i in sorted(idx)])
                                        else:
                                            concatenated=np.concatenate((np.flip(angles[i]),angles[j]))
                                            idx=np.unique(concatenated, return_index=True)[1]
                                            trial_dihedral = np.array(([concatenated[i] for i in sorted(idx)]))
                                            trial_dihedral = np.array(([trial_dihedral[0],trial_dihedral[1]], [trial_dihedral[1],trial_dihedral[2]], [trial_dihedral[2],trial_dihedral[3]]))
                                            if isSubset(trial_dihedral, bonds) == True:
                                                dihedrals.append([concatenated[i] for i in sorted(idx)])
                                            else:
                                                concatenated=np.concatenate((angles[i],np.flip(angles[j])))
                                                idx=np.unique(concatenated, return_index=True)[1]
                                                trial_dihedral = np.array(([concatenated[item3] for item3 in sorted(idx)]))
                                                dihedrals.append([concatenated[i] for i in sorted(idx)])
                                    ### This bit finds the improper torsions        
                                    if np.count_nonzero(angles[i]-angles[j] == 0) == 2:
                                        if np.nonzero(np.in1d(angles[i],angles[j]))[0][0] == 0:
                                            dihedrals.append([angles[i][0],angles[i][2],angles[j][2],angles[i][1]])

            dihedrals=np.asarray(dihedrals, dtype=int)            
            self.dihedrals = dihedrals    
        
        
        #self.bonds = bonds
        self.bonds = np.hstack((bonds, np.zeros((len(bonds),2), dtype=int)), dtype=int)
        #self.angles = angles
        self.angles = np.hstack((angles, np.zeros((len(angles),1), dtype=int)), dtype=int)
            
        try:    
            if len(bonds) >= 2 and len(angles) >= 2:
                self.int = np.vstack((self.bonds,self.angles,self.dihedrals))
            if len(bonds) >= 2 and len(angles) < 2:
                self.int = np.vstack((self.bonds,self.angles))
            if len(bonds) < 2 and len(bonds) >= 1:
                self.int = self.bonds
        except ValueError:
            print("Ill-defined internal coordinates")

def get_covalent_radius(atom):
    for i in atom_data:     
        if str(i[0]) == atom or i[1] == atom.capitalize(): 
            return i[5]
    raise KeyError("Element is not implemented.")

def get_connectivity(bonds, coord):
    fragments=[]
    test=bonds
    fragments.extend((test[0][0],test[0][1]))
    test=np.delete(test, 0, axis=0)
    test1=[]
    all_fragments=[]
    while len(test) != 0:
      if len(test1) != len(test):
        rows_to_delete=[]
        test1=test
        for i in range(len(test)):
          if np.in1d(fragments,test[i]).any() == True:
            fragments = np.unique(np.concatenate((fragments,test[i])))
            rows_to_delete.append(i)
        test = np.delete(test, rows_to_delete, axis=0)
      if len(test1) == len(test) or len(test) == 0:
        all_fragments.append(fragments)
        fragments = []
        if len(test) != 0:
          fragments.extend((test[0][0],test[0][1]))
          test=np.delete(test, 0, axis=0)
          test1=[]
          if len(test) == 0:
              all_fragments.append(fragments)

    new_bonds=[]
    for i in range(len(all_fragments)):
        for j in range(len(all_fragments)):
            if i != j and i < j:
                temporary_storage=[]
                all_intramolecular_bonds = np.stack(np.meshgrid(all_fragments[i], all_fragments[j]), -1).reshape(-1, 2)
                for k in range(len(all_intramolecular_bonds)):
                    bond_length = gf_method.length(coord[all_intramolecular_bonds[k][0]-1], coord[all_intramolecular_bonds[k][1]-1])
                    temporary_storage.append((bond_length, all_intramolecular_bonds[k][0], all_intramolecular_bonds[k][1]))
                new_bonds.append(min(temporary_storage)[1:])

    return new_bonds


def isSubset(input,bonds):
    input=np.sort(input)
    for i in range(len(input)):
        found = False
        for j in range(len(bonds)):
            if np.array_equal(input[i],bonds[j]) == True:
                found = True
                break
        if not found:
            return False
    return True


atom_data = [
    # atomic number, symbols, names, masses, bohr radius, covalent radius
    [  0, "X", "X",            0.000000, 0.000, 0*pm_to_bohr],  # 0
    [  1, "H", "Hydrogen",     1.007940, 0.324, 31*pm_to_bohr],  # 1
    [  2, "He", "Helium",      4.002602, 0.000, 28*pm_to_bohr],  # 2
    [  3, "Li", "Lithium",     6.941000, 1.271, 128*pm_to_bohr],  # 3
    [  4, "Be", "Beryllium",   9.012182, 0.927, 96*pm_to_bohr],  # 4
    [  5, "B", "Boron",       10.811000, 0.874, 84*pm_to_bohr],  # 5
    [  6, "C", "Carbon",      12.010700, 0.759, 76*pm_to_bohr],  # 6
    [  7, "N", "Nitrogen",    14.006700, 0.706, 71*pm_to_bohr],  # 7
    [  8, "O", "Oxygen",      15.999400, 0.678, 66*pm_to_bohr],  # 8
    [  9, "F", "Fluorine",    18.998403, 0.568, 57*pm_to_bohr],  # 9
    [ 10, "Ne", "Neon",       20.179700, 0.000, 58*pm_to_bohr],  # 10
    [ 11, "Na", "Sodium",     22.989769, 1.672, 166*pm_to_bohr],  # 11
    [ 12, "Mg", "Magnesium",  24.305000, 1.358, 141*pm_to_bohr],  # 12
    [ 13, "Al", "Aluminium",  26.981539, 1.218, 121*pm_to_bohr],  # 13
    [ 14, "Si", "Silicon",    28.085500, 1.187, 111*pm_to_bohr],  # 14
    [ 15, "P", "Phosphorus",  30.973762, 1.105, 107*pm_to_bohr],  # 15
    [ 16, "S", "Sulfur",      32.065000, 1.045, 105*pm_to_bohr],  # 16
    [ 17, "Cl", "Chlorine",   35.453000, 1.006, 102*pm_to_bohr],  # 17
    [ 18, "Ar", "Argon",      39.948000, 0.000, 106*pm_to_bohr],  # 18
]

    
