import numpy as np
import os
import sys
from input_reader import gaussian_parser, orca_parser
from inertia_and_com import inertia_tensor 
import output
import thermo_new as thermo
import pg
import tools
import gf_method
from tabulate import tabulate
joule_to_hartree=229371044869059970
kb=1.380649E-23
bohr_to_ang=0.529177249

use_qrrho = True
use_gaussian = True
use_orca = False

class initialize_orca:   # This initializes the required information from the *.hess and *.engrad files into instance variables
    """
    The initialize class currently only support ORCA files. The required files for full operationality are the *.hess, *.engrad files, and a internal coordinate definitions file.
    Requires: filename  -> the parent filename of the ORCA files, as in filename.hess, and filename.engrad.
              internal coordinates -> Currently supplied via a textfile, but will be improved to be read automatically from ORCA output file etc.
    Returns: atoms, masses, coord, center-of-mass coordinates, Cartesian Hessian, matrix of atomic mass triplets, inertia eigenvalues and eigenvectors, rotational constants in cm-1, point group, rotational symmetry number.
    """
    def __init__(self, filename, mult, temp=298.15, pressure=1, omega_0=100):
        self.filename = filename
        self.mult = mult
        self.temp = temp
        self.pressure = pressure
        self.omega_0 = omega_0
        self.stationary = True ## This defaults to True, but if gradient is loaded, then the value is checked with criterion.
        self.proj_modes = 0
        parse = orca_parser(self.filename)
        self.int = parse.int
        self.internal_coordinates = "internal_coordinates.txt"
        
        try:
            self.atoms = parse.atoms
            self.masses = parse.masses
            self.coord = parse.coord
            self.mass_matrix = parse.mass_matrix
            self.hessian_cart = parse.hessian_cart
            self.com = np.array(self.coord) - tools.get_center_mass(self.coord, self.masses)
            self.Ieigval, self.Ieigvec, self.B = inertia_tensor(self.com, self.masses)
            self.principal_axis_coordinates = np.matmul(np.linalg.inv(self.Ieigvec),self.com.T).T
            self.pg = pg.PointGroup(self.coord, self.atoms, self.masses).get_point_group()
            self.sn = tools.get_symmetry_number(self.pg)
            self.rot_sym = thermo.rotsym(self.Ieigval)
        except FileNotFoundError:
            print("The *.hess file is not found.")
    
        try:
            self.bmat, self.cmat = gf_method.get_bmat_and_cmat(self.internal_coordinates, self.com)
            self.fval, self.fvec = gf_method.gf_freqs(self.hessian_cart, self.mass_matrix, self.bmat, self.rot_sym)
            self.freq = output.frequencies(self.fval)
        except:
            print("Something is wrong with the internal coordinate definitions.")

        try:
            self.gradient_cart = parse.gradient_cart
            self.fval_p, self.fvec_p = gf_method.gf_proj_freqs(self.hessian_cart, self.gradient_cart, self.mass_matrix, self.bmat, self.cmat, self.rot_sym)
            self.freq_p = output.frequencies(self.fval_p)
            if np.max(np.abs(self.gradient_cart)) > 0.001:
                self.stationary = False
                self.proj_modes = 1
                self.freq = output.frequencies(self.fval_p)
            else:
                self.stationary = True
                self.freq = output.frequencies(self.fval)
                if any(self.freq) < 0.0:
                    self.proj_modes = 1
                else:
                    self.proj_modes = 0
        except AttributeError:
                None
        try:
            self.thermochemistry = thermo.get_thermochemistry(self.freq, self.proj_modes, self.mult, self.masses, self.B, self.rot_sym, self.sn, use_qrrho, self.temp, self.pressure, self.omega_0)
            self.ZPE = self.thermochemistry[0]
            self.U = self.thermochemistry[1]
            self.H = self.thermochemistry[2]
            self.S = self.thermochemistry[3]
            self.G = self.thermochemistry[4]
        finally:
            print()
###################################################################################################################################

class initialize_gaussian:   # THIS GAUSSIAN TEST
    """
    The initialize class currently only support ORCA files. The required files for full operationality are the *.hess, *.engrad files, and a internal coordinate definitions file.
    Requires: filename  -> the parent filename of the ORCA files, as in filename.hess, and filename.engrad.
              internal coordinates -> Currently supplied via a textfile, but will be improved to be read automatically from ORCA output file etc.
    Returns: atoms, masses, coord, center-of-mass coordinates, Cartesian Hessian, matrix of atomic mass triplets, inertia eigenvalues and eigenvectors, rotational constants in cm-1, point group, rotational symmetry number.
    """
    def __init__(self, filename, temp=298.15, pressure=1, omega_0=100):
        self.filename = filename
        self.temp = temp
        self.pressure = pressure
        self.omega_0 = omega_0
        self.stationary = True ## This defaults to True, but if gradient is loaded, then the value is checked with criterion.
        self.proj_modes = 0

        try:
            parse = gaussian_parser(filename)
            self.atoms = parse.atoms
            self.mult = parse.mult
            self.masses = parse.masses
            self.coord = parse.coord
            self.energy = parse.energy
            self.int = tools.get_redundant_internals(self.coord, self.atoms).int
            self.internal_coordinates = "internal-coordinates.txt"
            self.com = self.coord - tools.get_center_mass(self.coord, self.masses)
            self.hessian_cart = parse.hessian_cart
            self.mass_matrix = parse.mass_matrix
            self.Ieigval, self.Ieigvec, self.B = inertia_tensor(self.com, self.masses)
            self.principal_axis_coordinates = np.matmul(np.linalg.inv(self.Ieigvec),self.com.T).T
            self.pg = pg.PointGroup(self.coord, self.atoms, self.masses).get_point_group()
            self.sn = tools.get_symmetry_number(self.pg)
            self.rot_sym = thermo.rotsym(self.Ieigval)
        except FileNotFoundError:
            print("The *.fchk file is not found.")
        try:
            self.bmat, self.cmat = gf_method.get_bmat_and_cmat(self.internal_coordinates, self.com)
            self.fval, self.fvec = gf_method.gf_freqs(self.hessian_cart, self.mass_matrix, self.bmat, self.rot_sym)
            self.freq = output.frequencies(self.fval)
        except:
            print("Something is wrong with the internal coordinate definitions.")
        try:
            self.gradient_cart = parse.gradient_cart
            self.fval_p, self.fvec_p = gf_method.gf_proj_freqs(self.hessian_cart, self.gradient_cart, self.mass_matrix, self.bmat, self.cmat, self.rot_sym)
            self.freq_p = output.frequencies(self.fval_p)
            if np.max(np.abs(self.gradient_cart)) > 0.001:
                self.stationary = False
                self.proj_modes = 1
                self.freq = output.frequencies(self.fval_p)
            else:
                self.stationary = True
                self.freq = output.frequencies(self.fval)
                if any(self.freq) < 0.0:
                    self.proj_modes = 1
                else:
                    self.proj_modes = 0
        except FileNotFoundError:
            print("The *.engrad file is not found.")
        try:
            self.thermochemistry = thermo.get_thermochemistry(self.freq, self.proj_modes, self.mult, self.masses, self.B, self.rot_sym, self.sn, use_qrrho, self.temp, self.pressure, self.omega_0)
            self.ZPE = self.thermochemistry[0]
            self.U = self.thermochemistry[1]
            self.H = self.thermochemistry[2]
            self.S = self.thermochemistry[3]
            self.G = self.thermochemistry[4]
        finally:
            None




##### MAIN PROGRAM STRUCTURE #####
if use_gaussian == True and use_orca == False:
    print("Gaussian interface chosen, if this is not intended, edit the use_gaussian and use_orca flags in the vtst.py file")

if use_gaussian == False and use_orca == True:
    print("ORCA interface chosen, if this is not intended, edit the use_gaussian and use_orca flags in the vtst.py file")

if use_gaussian == True and use_orca == True:
    print("Both Gaussian and ORCA interfaces chosen, please disable one in the vtst.py file")
    print("Exiting...")
    sys.exit()

#%%
if use_gaussian == True:
    file_list=np.array([])
    E_list=np.array([])
    U_list=np.array([])
    H_list=np.array([])
    S_list=np.array([])
    G_list=np.array([])
    stationary_list=np.array([])

    #f.write("filename"+"\t"+"E (Eh)"+"\t"+"U (Eh)"+"\t"+"H (Eh)"+"\t"+"S (Eh)"+"\t"+"G (Eh)"+"Stationary"+"\n")
    if len(sys.argv) == 1:
        print("VTST script running for all *.fchk files using default parameters: T=298.15K, p=1atm, qRRHO with omega=100 cm-1")
        for file in os.listdir('.'):
            if file.endswith(".fchk"):
                mol = initialize_gaussian(file)
                file_list = np.append(file_list, str(file))
                E_list = np.append(E_list, mol.energy)
                U_list = np.append(U_list, mol.U)
                H_list = np.append(H_list, mol.H)
                S_list = np.append(S_list, mol.S)
                G_list = np.append(G_list, mol.G)
                stationary_list = np.append(stationary_list, mol.stationary)
                #f.write(str(file)+"\t"+str(mol.energy)+"\t"+str(mol.U)+"\t"+str(mol.H)+"\t"+str(mol.S)+"\t"+str(mol.G)+"\t"+str(mol.stationary)+"\n")
    if len(sys.argv) > 1:
       
        if (len(sys.argv)) == 2:
            file = str(sys.argv[1])
            print("VTST script running for ",file," using default parameters: T=298.15K, p=1atm, qRRHO with omega=100 cm-1")
            mol = initialize_gaussian(file)
            file_list = np.append(file_list, str(file))
            E_list = np.append(E_list, mol.energy)
            U_list = np.append(U_list, mol.U)
            H_list = np.append(H_list, mol.H)
            S_list = np.append(S_list, mol.S)
            G_list = np.append(G_list, mol.G)
            stationary_list = np.append(stationary_list, mol.stationary)
            #f.write(str(file)+"\t"+str(mol.energy)+"\t"+str(mol.U)+"\t"+str(mol.H)+"\t"+str(mol.S)+"\t"+str(mol.G)+"\t"+str(mol.stationary)+"\n")
        
        if (len(sys.argv)) == 3:
            file = str(sys.argv[1])
            T = sys.argv[2]
            mol = initialize_gaussian(file,temp=T)
            file_list = np.append(file_list, str(file))
            E_list = np.append(E_list, mol.energy)
            U_list = np.append(U_list, mol.U)
            H_list = np.append(H_list, mol.H)
            S_list = np.append(S_list, mol.S)
            G_list = np.append(G_list, mol.G)
            stationary_list = np.append(stationary_list, mol.stationary)
            #f.write(str(file)+"\t"+str(mol.energy)+"\t"+str(mol.U)+"\t"+str(mol.H)+"\t"+str(mol.S)+"\t"+str(mol.G)+"\t"+str(mol.stationary)+"\n")

        if (len(sys.argv)) == 3:
            file = str(sys.argv[1])
            T = sys.argv[2]
            p = sys.argv[3]
            print("VTST script running for ",file," using: T=",T,"K, p=",p,"atm, qRRHO with omega=100 cm-1")
            mol = initialize_gaussian(file,temp=T,pressure=p)
            file_list = np.append(file_list, str(file))
            E_list = np.append(E_list, mol.energy)
            U_list = np.append(U_list, mol.U)
            H_list = np.append(H_list, mol.H)
            S_list = np.append(S_list, mol.S)
            G_list = np.append(G_list, mol.G)
            stationary_list = np.append(stationary_list, mol.stationary)
            #f.write(str(file)+"\t"+str(mol.energy)+"\t"+str(mol.U)+"\t"+str(mol.H)+"\t"+str(mol.S)+"\t"+str(mol.G)+"\t"+str(mol.stationary)+"\n")

        if (len(sys.argv)) == 4:
            file = str(sys.argv[1])
            T = sys.argv[2]
            p = sys.argv[3]
            omg = sys.argv[4]
            print("VTST script running for ",file," using: T=",T,"K, p=",p,"atm, qRRHO with omega=",omg,"cm-1")
            mol = initialize_gaussian(file,temp=T,pressure=p,omega_0=omg)
            file_list = np.append(file_list, str(file))
            E_list = np.append(E_list, mol.energy)
            U_list = np.append(U_list, mol.U)
            H_list = np.append(H_list, mol.H)
            S_list = np.append(S_list, mol.S)
            G_list = np.append(G_list, mol.G)
            stationary_list = np.append(stationary_list, mol.stationary)
            #f.write(str(file)+"\t"+str(mol.energy)+"\t"+str(mol.U)+"\t"+str(mol.H)+"\t"+str(mol.S)+"\t"+str(mol.G)+"\t"+str(mol.stationary)+"\n")

idx = file_list.argsort()
file_list = file_list[idx]
E_list = E_list[idx]
U_list=U_list[idx]
H_list=H_list[idx]
S_list=S_list[idx]
G_list=G_list[idx]
stationary_list=stationary_list[idx]
                
output_list = np.stack([file_list.astype(str),E_list.astype(float),U_list.astype(float),H_list.astype(float),S_list.astype(float),G_list.astype(float),stationary_list.astype(str)],axis=1)
print(tabulate(output_list,headers=["filename","E (Eh)","U (Eh)","H (Eh)","S (Eh/K)","G (Eh)","Stationary (True=1, False=0)"], tablefmt="rst", floatfmt=(".4f",".4f",".4f",".4f",".4f",".4f",".0f"), colalign=("left","left","left","left","left","left","left")))

with open("output-vtst.out", "w") as f:
    f.write(tabulate(output_list,headers=["filename","E (Eh)","U (Eh)","H (Eh)","S (Eh/K)","G (Eh)","Stationary (True=1, False=0)"], tablefmt="plain", floatfmt=(".10f",".10f",".10f",".10f",".10f",".10f",".0f"), colalign=("left","left","left","left","left","left","left")))
print("Output written also to output-vtst.out file")

