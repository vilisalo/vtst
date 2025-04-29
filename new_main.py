import numpy as np
import sys
from input_reader import gaussian_parser, orca_parser
from inertia_and_com import inertia_tensor 
import output
import thermo_new as thermo
import pg
import tools
import gf_method
joule_to_hartree=229371044869059970
kb=1.380649E-23
bohr_to_ang=0.529177249

use_qrrho = False

#### This currently only works for ORCA ####
class initialize:   # This initializes the required information from the *.hess and *.engrad files into instance variables
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
#%%
class initialize1:   # THIS GAUSSIAN TEST
    """
    The initialize class currently only support ORCA files. The required files for full operationality are the *.hess, *.engrad files, and a internal coordinate definitions file.
    Requires: filename  -> the parent filename of the ORCA files, as in filename.hess, and filename.engrad.
              internal coordinates -> Currently supplied via a textfile, but will be improved to be read automatically from ORCA output file etc.
    Returns: atoms, masses, coord, center-of-mass coordinates, Cartesian Hessian, matrix of atomic mass triplets, inertia eigenvalues and eigenvectors, rotational constants in cm-1, point group, rotational symmetry number.
    """
    def __init__(self, filename, temp=298.15, pressure=1, omega_0=100):
        self.filename = filename
        self.internal_coordinates = "internal_coordinates.txt"
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
            print()





#%%


temp=np.array([298.15])
pressure=1
mult=3


thermo.get_thermochemistry(freq, proj_modes, mult, mass, B, rot_sym, sn, qrrho)

if proj_modes == 0:
    print("Minimum energy point."," proj_modes = ",proj_modes)
if proj_modes == 1:
    print("Either reaction path or transition frequency projected in G calculation."," proj_modes = ",proj_modes)
if proj_modes >= 2:
    print("Two or more modes projected, make sure this is wanted."," proj_modes = ",proj_modes)

if frequencies[0] == freq[0]:
    print("Non-projected frequencies used for G.")
if frequencies[0] == freq_p[0]:
    print("Projected frequencies used for G.")

print("Gibbs energy corrections at various T: (call temp for list of temperatures)")
#print(G_corr)
#print(' '.join(map(str,np.round(G_corr,8)))," Eh")
#print(np.round(G_corr,8)," Eh")
print(' '.join(map(str,np.round(G_corr,8))))