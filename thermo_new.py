import numpy as np

### Constants ###
pi = np.pi
k=1.380649E-23 # J/K
h=6.62607015E-34 # Js
R=8.31446261815324 # J/(K.mol)
c=29979245800 # speed of light cm/s
NA=6.0221408E23 # avogadro's number 1/mol
joule_to_hartree=229371044869059970
bohr_to_meter=5.29177249E-11
R_kcal=0.0019872036

### Functions ###

def entropy(freq, temp, omega_0):
    """
    Function that calculates regular vibrational entropy, free rotor entropy, and 
    the QRRHO entropy.
    
    Requires : Vibrational frequencies, temperature, and omega_0 value for the
    QRRHO
    
    Returns : Vibrational entropy (=[0]), Free-rotor entropy (=[1]), and QRRHO entropy (=[2])
    """
    B_av=1E-44 # kg/m^2    Could be changed to isotropically averaged moment of inertia in the future?
    beta = 1/(k*temp) # Thermodynamic beta
    
    # Weighting function for balancing between Sv and Sr in QRRHO#
    def w(input):
        #omega_0=100 # cm-1
        return(1/(1+(omega_0/input)**4))
    
    # Harmonic oscillator entropy
    Sv = k*((beta*h*c*freq)/(np.exp(beta*h*c*freq)-1)-np.log(1-np.exp(-beta*h*c*freq)))*joule_to_hartree
    # Free-rotor moment-of-inertia for corresponding wavenumber in cm-1 #
    mu = h/(8*pi**2*c*freq)
    mu_eff = mu*B_av/(mu+B_av)
    # Free-rotor entropy
    Sr = k*(1/2+np.log(np.sqrt((8*pi**3*mu_eff*k*temp)/(h**2))))*joule_to_hartree
    # qRRHO vibrational entropy
    S = w(freq)*Sv+(1-w(freq))*Sr
    return(Sv, Sr, S)

def vib_temp(freq):
    """
    Function that calculates the vibrational temperature for given vibrational frequency
    
    Requires : Vibrational frequencies (cm-1)
    
    Returns : Vibrational temperature (K)
    """
    return ((h*freq*c)/k)

def rot_temp(B):
    """
    Function that calculates the rotational temperature for given rotational constant
    
    Requires : Rotational constants (cm-1)
    
    Returns : Rotational temperature (K)
    """
    return((h*c*B)/k)

def rotsym(Ieigval):
    """
    Function that defines the type of rigid rotor, this information is used when deciding,
    which rotational entropy equation is used.
    
    Requires : Moment of inertia eigenvalues (Ieigval), sorted from smallest to largest
    
    Returns: Type of rotor (rot_sym), 0 = Spherical top, 1 = Linear rotor, 2 = Oblate symmetric top, 3 = Prolate symmetric top, 4 = Asymmetric rotor.
    """
    if round(Ieigval[0],24)==round(Ieigval[1],24)==round(Ieigval[2],24):
        rot_sym=0   # Spherical top
        
    if round((Ieigval[1]-Ieigval[0])/Ieigval[1],8)==1 and round(Ieigval[1],24)==round(Ieigval[2],24):
        rot_sym=1   # Linear rotor
        
    if round(Ieigval[0],24)==round(Ieigval[1],24) and round(Ieigval[1],24)<round(Ieigval[2],24):
        rot_sym=2   # Oblate symmetric top
        
    if Ieigval[0]<Ieigval[1] and round((Ieigval[1]-Ieigval[0])/Ieigval[1],8)!=1 and round(Ieigval[1],24)==round(Ieigval[2],24):
        rot_sym=3   # Prolate symmetric top
        
    if round(Ieigval[0],24)!=round(Ieigval[1],24)!=round(Ieigval[2],24):
        rot_sym=4   # Asymmetric top
    return rot_sym

def ZPE(freq,proj_modes):
    """
    Function that calculates the zero-point vibrational energies.
    
    Requires : Vibrational frequencies (cm-1) and the number of projected modes (proj_modes); these are not included in the ZPE.
    
    Returns : Zero-point vibrational energy, ZPE
    """
    ZPE=0
    for i in range(proj_modes,len(freq),1):
        ZPE+=freq[i]*c*h*(1/2)
    ZPE=ZPE*joule_to_hartree
    return ZPE

def TVE(freq,proj_modes,temp,omega_0):
    """
    Function that calculates the thermal vibrational energies. Also calculates the QRRHO corrected energies
    
    Requires : Vibrational frequencies (cm-1),number of projected modes (proj_modes), temperature, and omega_0 value for QRRHO.
    
    Returns : Thermal vibrational energy, TVE (=[0]), and QRRHO-corrected TVE (=[1])
    """
    HO=FR=TVE=TVE_quasi=0
    def w(input):
        return(1/(1+(omega_0/input)**4))
    for i in range(proj_modes,len(freq),1):
        HO=k*vib_temp(freq[i])/(np.exp(vib_temp(freq[i])/temp)-1)
        FR=1/2*k*temp
        TVE += HO
        TVE_quasi += w(freq[i])*HO + (1-w(freq[i]))*FR
    TVE=TVE*joule_to_hartree
    TVE_quasi=TVE_quasi*joule_to_hartree
    return(TVE, TVE_quasi)                                                    # internal thermal energy of free rotor
        
def TRE(temp,rot_sym):
    """
    Function that calculates the thermal rigid-rotor energies. The used equation depends on whether the molecule is linear (rot_sym=1), or non-linear (rot_sym =! 1)
    
    Requires : Temperature (K), and rotor type (rot_sym)
    
    Returns : Thermal rigid-rotor rotational energy (Eh)
    """
    if rot_sym == 1:
        TRE=k*temp*joule_to_hartree
    else:
        TRE=3/2*k*temp*joule_to_hartree
    return TRE

def TTE(temp):
    """
    Function that calculates thermal translational energy.
    
    Requires : Temperature (K)
    
    Returns : Thermal translational energy (Eh)
    """
    TTE=3/2*k*temp*joule_to_hartree
    return TTE

def S_t(temp,mass,pressure=1):
    """
    Function that calculates translational entropy with the Sackur-Tetrode -equation (ideal gas assumption).
    
    Requires : Temperature (K), Pressure (atm), and the mass vector of the molecular system (amu)
    
    Returns : Translational entropy (Eh K-1)
    """
    pressure_Pa=pressure*101325
    M=np.sum(mass)/1000
    S_t=k*(5/2+np.log(((2*pi*M*k*temp)/(NA*h**2))**(3/2)*(k*temp)/pressure_Pa))*joule_to_hartree
    return S_t

def S_v(freq,proj_modes,temp,omega_0=100):
    """
    Function that calculates vibrational entropy. Both the usual harmonic oscillator entropy, as well as the QRRHO-corrected vibrational entropy.
    
    Requires : Vibrational frequencies (cm-1), number of projected modes, temperature (K), omega_0 value for the QRRHO.
    
    Returns : Vibrational entropy (=[0]), and the QRRHO-corrected vibrational entropy (=[1]), (Eh K-1)
    """
    S_v=S_v_quasi=0
    for i in range(proj_modes,len(freq),1):
        S_v+=entropy(freq[i],temp, omega_0)[0]
        S_v_quasi+=entropy(freq[i],temp, omega_0)[2]
    return(S_v, S_v_quasi)

def enthalpy(U_corr, temp):
    """
    Function that calculates total enthalpy correction, for single ideal gas molecule.
    
    Requires : Total correction to internal energy (both ZPE and thermal corrections), U_corr, and temperature (K)
    
    Returns : Total enthalpy correction H_corr (Eh)
    """
    H_corr = U_corr + k*temp*joule_to_hartree
    return H_corr

def S_e(mult):
    """
    Function that calculates the electronic entropy. Only the ground-state is assumed to affect the entropy.
    
    Requires : Spin multiplicity
    
    Returns : Electronic entropy (Eh K-1)
    """
    S_e = k*np.log(mult)*joule_to_hartree
    return S_e

def S_r(rot_sym,temp,B,sn):
    """
    Function that calculates the rigid-rotor rotational entropy.
    
    Requires : Type of the rotor (rot_sym), temperature (K), rotational constants (cm-1), rotational symmetry number (sn).
    
    Returns : Rigid-rotor rotational entropy (Eh K-1)
    """
    if rot_sym == 1:
        S_r=k*(np.log(temp/(sn*rot_temp(B[1])))+1)*joule_to_hartree
    if rot_sym != 1:
        S_r=k*(np.log((1/sn)*np.sqrt((np.pi*temp**3)/(rot_temp(B[0])*rot_temp(B[1])*rot_temp(B[2]))))+3/2)*joule_to_hartree
    return S_r

def get_thermochemistry(freq,proj_modes,mult,mass,B,rot_sym,sn,qrrho,temp=298.15,pressure=1,omega_0=100):
    print("Thermochemistry output at ",temp,"K temperature and ",pressure,"atm pressure.")
    
    ### Non-thermal contributions: ###
    zpe_value = ZPE(freq,proj_modes)
    ### Thermal contributions: ###
    tte = TTE(temp)
    tre = TRE(temp, rot_sym)
    if qrrho == True:
        if omega_0 == 100:
            print("The default value of omega_0 = 100 cm-1 is used.")
        else:
            print("The omega_0 = ",omega_0,"cm-1.")
        tve = TVE(freq, proj_modes, temp, omega_0)[1]
        s_v = S_v(freq, proj_modes, temp, omega_0)[1]
    else:
        tve = TVE(freq, proj_modes, temp, omega_0)[0]
        s_v = S_v(freq, proj_modes, temp, omega_0)[0]
    
    ### Total correction to interal energy: ###
    U_corr = zpe_value + tte + tre + tve
    ### Total correction to enthalpy: ###
    H_corr = enthalpy(U_corr, temp)
    
    ### Remaining corrections to entropy: ###
    s_t = S_t(temp, mass, pressure)
    s_r = S_r(rot_sym, temp, B, sn)
    s_e = S_e(mult)
    S_corr = s_e + s_t + s_r + s_v
    
    ### Gibbs energy correction: ###
    G_corr = H_corr - temp*S_corr
    return (zpe_value, U_corr, H_corr, S_corr, G_corr)
    
    