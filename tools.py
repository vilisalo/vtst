import warnings
import numpy as np

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
