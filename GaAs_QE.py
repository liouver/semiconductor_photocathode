'''
Created on Feb 2018 by W. Liu
Monte Carlo methode to simulate the photoemission based on three-step model:
photoexcited, transportation and emission
|----------------------------|
|                            |
|                            |
|                            |    ^y
|                            |    |
|----------------------------|    |
------------------------------>z
electron distribution: (z,y,vz,vy,v,E) in GaAs
z direction: exponential distribution for photoexcited electrons
y direction: Gauss distribution for photoexcited electrons (depend on laser)
'''
import random
import numpy as np
from scipy import integrate
from scipy.stats import expon, maxwell
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

start_time = time.time()

# ----Define fundamental constants and general parameters----
pi = np.pi
two_pi = 2 * pi
eps0 = 8.85419e-12  # F/m, vacuum dielectric constant
eps = 12.9 * eps0  # F/m, dielectric constant for low frequency
eps_high = 10.89 * eps0  # F/m, for high frequency
kB = 1.38066e-23  # J/K, Boltzmann constant
ec = 1.60219e-19  # C
h_ = 1.05459e-34  # J*s, Planc constant
c = 2.99792e8  # m/s, light speed
m_e = 9.109e-31  # kg, electron mass
m_hh = 0.5 * m_e  # effective heavy hole mass
m_lh = 0.076 * m_e  # effective light hole mass
m_so = 0.145 * m_e  # effective split-off band mass
m_T = 0.063 * m_e  # Gamma valley effective electron mass
m_L = 0.555 * m_e  # L valley effective electron mass
m_X = 0.851 * m_e  # X valley effective electron mass
m_h = (m_hh**1.5 + m_lh**1.5 + m_so**1.5)**(2 / 3)

# ----Set material parameters----
T = 298  # K, material temperature
N_A = 1e25  # m**(-3), doping concentration
rou = 5.32e3  # kg/m**3, density of GaAs
E_T = kB * T / ec
Eg = 1.519 - 0.54 * 10**(-3) * T**2 / (T + 204)  # eV, bandgap
# Tiwari, S, Appl. Phys. Lett. 56, 6 (1990) 563-565. (experiment data)
# Eg = Eg - 2 * 10**(-11) * np.sqrt(N_A)
DEg = 3 * ec / 16 / pi / eps * np.sqrt(ec**2 * N_A / eps / kB / T)
Eg = Eg - DEg
DE = 0.34  # eV, split-off energy gap
# E_B = Eg / 3  # only for NA = 10**19 cm**-3
EB_data = np.genfromtxt('GaAs_Band_Bending.csv', delimiter=',')
func0 = interp1d(EB_data[:, 0] * 1e6, EB_data[:, 1])
E_B = func0(N_A)
W_B = np.sqrt(2 * eps * E_B / ec / N_A) * 10**9  # nm
# print(Eg, E_B, W_B, DEg, E_T)
E_A = -0.1  # eV, electron affinity
thick = 1e4  # nm, thickness of GaAs active layer
surface = 0  # position of electron emission, z = 0

# ----Define simulation time, time step and total photon number----
total_time = 10e-12  # s
step_time = 1e-14  # s
Ni = 100000  # incident photon number

# ----Set the electrical field----
field_y = 0
field_z = 1e3  # V/m
E_sch = 0  # eV, vacuum level reduction by Schottky effect

# ----Set parameters for acoustic phonon scattering----
V_T = 7.01  # eV, acoustic deformation potential for Gamma valley
V_L = 9.2  # eV,  for L valley
V_X = 9.0  # eV, for X valley
ul = 5.24e3  # m/s, longitudial sound speed
alpha_T = 0.61  # 1/eV, nonparabolicity factor for Gamma valley
alpha_L = 0.461  # 1/eV, for L valley
alpha_X = 0.204  # 1/eV, for X valley

# ------ set parameters for optical phonon scattering
num_T_valley = 1  # equivalent valleys for Gamma valley
num_L_valley = 4  # for L valley
num_X_valley = 3  # for X valley
E_L_T = 0.29  # eV, splitting energy between Gamma and L valley
E_X_T = 0.48  # eV, splitting energy between Gamma and X valley
D_T_L = 1.8e10  # eV/m, optical coupling constant for between G and L valley
D_T_X = 10e10  # eV/m, between Gamma to X valley
D_L_L = 5e10  # eV/m, between L and L valley
D_L_X = 1e10  # eV/m, between L and X valley
D_X_X = 10e10  # eV/m, between X and X valley
phonon_T_L = 0.0278  # eV, optical phonon energy between G and L valley
phonon_T_X = 0.0299
phonon_L_L = 0.029
phonon_L_X = 0.0293
phonon_X_X = 0.0299


def photon_to_electron(hw):
    ''' electrons in valence band aborption photon to excited to conduction
    band. Only consider these electrons in the heavy hole, light hole and
    split-off band would be excited, and can only excited into Gamma valley.
    Given photon energy, return excited electron energy. '''
    # nonparabolicity factor, 1/eV
    # alpha_T = 0.58 + (T - 77) * (0.61 - 0.58) / (300 - 77)
    Ei = random.uniform(Eg, hw - 0.01)
    if Ei >= Eg + DE:
        x = random.randint(1, 6)
        if x in [1, 2, 3]:  # heavy hole
            E1 = hw - Ei
            Gamma = 1 + m_hh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_hh / m_T)
            E_e = E1 - DE_h
        elif x == 4:  # light hole
            E1 = hw - Ei
            Gamma = 1 + m_lh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_lh / m_T)
            E_e = E1 - DE_h
        elif x in [5, 6]:  # split-off band
            E1 = hw - Ei
            Gamma = 1 + m_so / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_so / m_T)
            E_e = E1 - DE_h
    elif Eg <= Ei < Eg + DE:
        x = random.randint(1, 4)
        if x in [1, 2, 3]:  # heavy hole
            E1 = hw - Ei
            Gamma = 1 + m_hh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_hh / m_T)
            E_e = E1 - DE_h
        elif x == 4:  # light hole
            E1 = hw - Ei
            Gamma = 1 + m_lh / m_e + 2 * alpha_T * E1
            DE_h = (1 - np.sqrt(1 - 4 * alpha_T * E1 * (1 + alpha_T * E1) /
                                Gamma**2)) / (2 * alpha_T) / Gamma
            # DE_h = E1 / (1 + m_lh / m_T)
            E_e = E1 - DE_h
    else:
        E_e = 0.02
    # print(DE_h, E_e)

    return E_e


def electron_distribution(hw, types):
    '''photon decay as exponential distribution in GaAs,
    generating electrons with exponential distribution in z direction.
    Given photon energy, photon number and sample thickness.
    Return excited electron position(z,y),direction(vz,vy),velocity and energy
    '''
    energy = []
    if types == 1:  # Four bands photoexcited
        for i in range(Ni):
            energy.append(photon_to_electron(hw))
        energy = np.array(energy)
    elif types == 2:  # use density of state from reference
        DOS = np.genfromtxt('DOS.csv', delimiter=',')
        func1 = interp1d(DOS[:, 0], DOS[:, 1])
        '''
        fig, ax = plt.subplots()
        e = np.linspace(-2.8, 3, 100)
        ax.plot(e, func1(e))
        plt.show()'''
        E0 = Eg
        norm, err = integrate.quad(lambda e: func1(e - hw) * func1(e), E0, hw,
                                   limit=10000)
        Ei = np.linspace(E0, hw, int((hw - E0) / 0.001))
        for i in range(len(Ei)):
            num = 1.5 * Ni * func1(Ei[i]) * func1(Ei[i] - hw) * 0.001 / norm
            E_num = np.empty(int(num))
            E_num.fill(Ei[i] - E0)
            energy.extend(E_num)
        np.random.shuffle(energy)
    else:
        print('Wrong photon-to-electron type')

    absorb_data = np.genfromtxt('absorp_coeff_GaAs.txt', delimiter=',')
    func2 = interp1d(absorb_data[:, 0], absorb_data[:, 1])
    alpha = func2(hw) * 10**-7  # 1/nm,  absorption coefficient
    # photon distribution in GaAs (exp distribution random variables)
    z_exp = expon.rvs(loc=0, scale=1 / alpha, size=Ni)
    # photon (electron) distribution in GaAs with thickness less than thick
    z_pos = [z for z in z_exp if z <= thick]
    z_pos = np.array(z_pos)
    Num = len(z_pos)
    energy = np.resize(energy, Num)
    # y axis, position set to gauss distribution
    y_pos = np.random.normal(0, 0.25e6, Num)
    velocity = np.sqrt(2 * np.abs(energy) * ec / m_T) * 10**9
    # Isotropic distribution, phi=2*pi*r, cos(theta)=1-2*r
    r = np.random.uniform(0, 1, Num)
    phi = two_pi * r
    theta = np.arccos(1 - 2 * r)
    vz = velocity * np.cos(theta)
    vy = velocity * np.sin(theta) * np.sin(phi)
    distribution_2D = np.vstack((z_pos, y_pos, vz, vy, velocity, energy)).T

    '''
    distribution_2D = []
    for i in range(len(z_pos)):
        # initial angle between the projection on XY surface and y axis
        phi = random.uniform(0, 2 * pi)
        # initial angle between the direction and z axis
        theta = random.uniform(0, 2 * pi)
        # y_pos = random.uniform(-1 * 10**6, 1 * 10**6)
        y_pos = random.gauss(0, 0.25 * 10**6)
        velocity = np.sqrt(2 * np.abs(energy[i]) * ec / m_T) * 10**9  # nm/s
        vz = velocity * np.cos(theta)
        vy = velocity * np.sin(theta) * np.cos(phi)
        distribution_2D.append([z_pos[i], y_pos, vz, vy, velocity, energy[i]])
    distribution_2D = np.array(distribution_2D)  # ([z, y, vz, vy, v, E])'''
    '''
    plt.figure()
    plt.hist(distribution_2D[:, 5], bins=100)
    plt.show()
    '''
    return distribution_2D


def impurity_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0.001)
    if types == 1:
        k_e = np.sqrt(2 * m_T * energy * ec) / h_  # 1/m, wavevector
        n_i = N_A  # m**-3, impurity concentration
        # T_e = np.mean(energy) * ec / kB
        a2 = (eps * kB * T) / (n_i * ec**2)  # m**2
        # print(4 * a2 * k_e**2)
        # e-impurity scattering rate, (Y. Nishimura, Jnp. J. Appl. Phys.)
        Rate_ei = (n_i * ec**4 * m_T) / (8 * pi * eps**2 * h_**3 * k_e**3) \
            * (np.log(1 + 4 * a2 * k_e**2) -
               (4 * a2 * k_e**2) / (1 + 4 * a2 * k_e**2))
    elif types == 2:
        k_e = np.sqrt(2 * m_L * energy * ec) / h_  # 1/m, wavevector
        n_i = N_A  # m**-3, impurity concentration
        a2 = (eps * kB * T) / (n_i * ec**2)  # m**2
        # print(4 * a2 * k_e**2)
        # e-impurity scattering rate, (Y. Nishimura, Jnp. J. Appl. Phys.)
        Rate_ei = (n_i * ec**4 * m_L) / (8 * pi * eps**2 * h_**3 * k_e**3) \
            * (np.log(1 + 4 * a2 * k_e**2) -
               (4 * a2 * k_e**2) / (1 + 4 * a2 * k_e**2))
    elif types == 3:
        k_e = np.sqrt(2 * m_X * energy * ec) / h_  # 1/m, wavevector
        n_i = N_A  # m**-3, impurity concentration
        a2 = (eps * kB * T) / (n_i * ec**2)  # m**2
        # e-impurity scattering rate, (Y. Nishimura, Jnp. J. Appl. Phys.)
        Rate_ei = (n_i * ec**4 * m_X) / (8 * pi * eps**2 * h_**3 * k_e**3) \
            * (np.log(1 + 4 * a2 * k_e**2) -
               (4 * a2 * k_e**2) / (1 + 4 * a2 * k_e**2))
    else:
        print('Wrong electron impurity scattering')
    return Rate_ei


def electron_hole_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    energy = energy.clip(0.001)
    n_h = 0.5 * N_A  # m**-3, hole concentration
    T_h = 298  # K, hole temperature
    T_e = np.mean(energy) * ec / kB  # K, electron temperature
    beta2 = n_h * ec**2 / eps / kB * (1 / T_e + 1 / T_h)
    if types == 1:
        b = 8 * m_T * energy * ec / h_**2 / beta2
        Rate_eh = n_h * ec**4 / 16 / 2**0.5 / pi / eps**2 / m_T**0.5 / \
            energy**1.5 / ec**1.5 * (np.log(1 + b) - b / (1 + b))
    elif types == 2:
        b = 8 * m_L * energy * ec / h_**2 / beta2
        Rate_eh = n_h * ec**4 / 16 / 2**0.5 / pi / eps**2 / m_L**0.5 / \
            energy**1.5 / ec**1.5 * (np.log(1 + b) - b / (1 + b))
    elif types == 3:
        b = 8 * m_X * energy * ec / h_**2 / beta2
        Rate_eh = n_h * ec**4 / 16 / 2**0.5 / pi / eps**2 / m_X**0.5 / \
            energy**1.5 / ec**1.5 * (np.log(1 + b) - b / (1 + b))
    else:
        print('Wrong electron-hole scattering type')
    return Rate_eh


def carrier_scattering(electron_energy, hole_energy):
    ''' PRB 36, 6018 (2017) '''
    n = len(electron_energy)
    # electron_energy = electron_energy.clip(0.001)
    k_e = np.sqrt(2 * m_T * electron_energy * ec) / h_  # 1/m, wavevector
    k_h = np.sqrt(2 * m_h * hole_energy * ec) / h_
    T_e = np.mean(electron_energy) * ec / kB
    T_h = np.mean(hole_energy) * ec / kB
    # n0 = N_A  # hole number
    # n_e = N_A  # electron number
    n_h = 0.1 * N_A  # m**-3, hole concentration
    mu = m_T * m_hh / (m_T + m_hh)
    beta2 = n_h * ec**2 / eps / kB * (1 / T_e + 1 / T_h)
    # print(beta2)
    Rate_eh = []
    # Rate_he = []
    # Rate_ee = []
    # Rate_hh = []
    ke = np.linspace(min(k_e), max(k_e), 100)
    # print(electron_energy, 4 * ke**2 / beta2)
    # kh = np.linspace(min(k_h), 1. * max(k_h), 100)
    for i in range(len(ke)):
        Q_eh = 2 * mu * np.abs(ke[i] / m_T - k_h / m_h)
        # Q_he = 2 * mu * np.abs(kh[i] / m_h - k_e / m_T)
        Rate1 = n_h * mu * ec**4 / two_pi / eps**2 / h_**3 / n * \
            sum(Q_eh / beta2 / (Q_eh**2 + beta2))
        '''Rate2 = n_e * mu * ec**4 / two_pi / eps**2 / h_**3 / n * \
            sum(Q_he / beta2 / (Q_he**2 + beta2))
        Rate3 = n_e * m_T * ec**4 / 4 / pi / eps**2 / h_**3 / n * \
            sum(np.abs(ke[i] - k_e) / beta2 / ((ke[i] - k_e)**2 + beta2))
        Rate4 = n_h * m_h * ec**4 / 4 / pi / eps**2 / h_**3 / n * \
            sum(np.abs(kh[i] - k_h) / beta2 / ((kh[i] - k_h)**2 + beta2))'''
        Rate_eh.append(Rate1)
        # Rate_he.append(Rate2)
        # Rate_ee.append(Rate3)
        # Rate_hh.append(Rate4)
    Rate_eh = np.array(Rate_eh)
    # Rate_he = np.array(Rate_he)
    # Rate_ee = np.array(Rate_ee)
    # Rate_hh = np.array(Rate_hh)
    func_eh = interp1d(ke, Rate_eh)
    Rate_eh = func_eh(k_e)
    '''
    fig, ax = plt.subplots()
    ax.semilogy(ke, Rate_eh, '.', kh, Rate_he, '.', ke, Rate_ee, '.')
    ax.set_xlabel(r'Electron energy (eV)', fontsize=14)
    ax.set_ylabel(r'scattering rate ($s^{-1}$)', fontsize=14)
    plt.tight_layout()
    plt.show()'''
    return Rate_eh


def acoustic_phonon_scattering(energy, types):
    '''
    types=1, scattering for Gamma valley
    types=2, scattering for L valley
    types=3, scattering for X valley
    '''
    if types == 1:
        Rate_ac = 2**0.5 * m_T**1.5 * kB * T * V_T**2 * energy**0.5 *\
            ec**2.5 / pi / h_**4 / ul**2 / rou *\
            (1 + 2 * alpha_T * energy) *\
            (1 + alpha_T * energy)**0.5
    elif types == 2:
        Rate_ac = 2**0.5 * m_L**1.5 * kB * T * V_L**2 * energy**0.5 *\
            ec**2.5 / pi / h_**4 / ul**2 / rou *\
            (1 + 2 * alpha_L * energy) *\
            (1 + alpha_L * energy)**0.5
    elif types == 3:
        Rate_ac = 2**0.5 * m_X**1.5 * kB * T * V_X**2 * energy**0.5 *\
            ec**2.5 / pi / h_**4 / ul**2 / rou *\
            (1 + 2 * alpha_X * energy) *\
            (1 + alpha_X * energy)**0.5
    else:
        print('Wrong acoustic phonon scattering type')
    return Rate_ac


def optical_phonon_scattering(energy, types):
    '''
    types=1, scattering for Gamma to L valley
    types=2, scattering for Gamma to X valley
    types=3, scattering for L to Gamma valley
    types=4, scattering for L to L valley
    types=5, scattering for L to X valley
    types=6, scattering for X to Gamma valley
    types=7, scattering for X to L valley
    types=8, scattering for X to X valley
    '''
    if types == 1:
        N_op = kB * T / ec / phonon_T_L
        tempEnergy = (energy + phonon_T_L - E_L_T).clip(0)
        Rate_ab = D_T_L**2 * m_T**1.5 * num_L_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_L - E_L_T).clip(0)
        Rate_em = D_T_L**2 * m_T**1.5 * num_L_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
    elif types == 2:
        N_op = kB * T / ec / phonon_T_X
        tempEnergy = (energy + phonon_T_X - E_X_T).clip(0)
        Rate_ab = D_T_X**2 * m_T**1.5 * num_X_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_X - E_X_T).clip(0)
        Rate_em = D_T_X**2 * m_T**1.5 * num_X_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_T * tempEnergy) *\
            (1 + alpha_T * tempEnergy)**0.5
    elif types == 3:
        N_op = kB * T / ec / phonon_T_L
        tempEnergy = (energy + phonon_T_L + E_L_T).clip(0)
        Rate_ab = D_T_L**2 * m_L**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_L + E_L_T).clip(0)
        Rate_em = D_T_L**2 * m_L**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
    elif types == 4:
        N_op = kB * T / ec / phonon_L_L
        tempEnergy = (energy + phonon_L_L).clip(0)
        Rate_ab = D_L_L**2 * m_L**1.5 * (num_L_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_L * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
        tempEnergy = (energy - phonon_L_L).clip(0)
        Rate_em = D_L_L**2 * m_L**1.5 * (num_L_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
    elif types == 5:
        N_op = kB * T / ec / phonon_L_X
        tempEnergy = (energy + phonon_L_X + E_L_T - E_X_T).clip(0)
        Rate_ab = D_L_X**2 * m_L**1.5 * num_X_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
        tempEnergy = (energy - phonon_L_X + E_L_T - E_X_T).clip(0)
        Rate_em = D_L_X**2 * m_L**1.5 * num_X_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_L * tempEnergy) *\
            (1 + alpha_L * tempEnergy)**0.5
    elif types == 6:
        N_op = kB * T / ec / phonon_T_X
        tempEnergy = (energy + phonon_T_X + E_X_T).clip(0)
        Rate_ab = D_T_X**2 * m_X**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
        tempEnergy = (energy - phonon_T_X + E_X_T).clip(0)
        Rate_em = D_T_X**2 * m_X**1.5 * num_T_valley / 2**0.5 / pi / rou /\
            h_**2 / phonon_T_L * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
    elif types == 7:
        N_op = kB * T / ec / phonon_L_X
        tempEnergy = (energy + phonon_L_X - E_L_T + E_X_T).clip(0)
        Rate_ab = D_L_X**2 * m_X**1.5 * num_L_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
        tempEnergy = (energy - phonon_L_X - E_L_T + E_X_T).clip(0)
        Rate_em = D_L_X**2 * m_X**1.5 * num_L_valley / 2**0.5 /\
            pi / rou / h_**2 / phonon_L_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
    elif types == 8:
        N_op = kB * T / ec / phonon_X_X
        tempEnergy = (energy + phonon_X_X).clip(0)
        Rate_ab = D_X_X**2 * m_X**1.5 * (num_X_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_X_X * N_op * ec * \
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
        tempEnergy = (energy - phonon_X_X).clip(0)
        Rate_em = D_X_X**2 * m_X**1.5 * (num_X_valley - 1) / 2**0.5 /\
            pi / rou / h_**2 / phonon_X_X * (N_op + 1) * ec *\
            np.sqrt(ec * tempEnergy) *\
            (1 + 2 * alpha_X * tempEnergy) *\
            (1 + alpha_X * tempEnergy)**0.5
    else:
        print('Wrong optical phonon scattering type')
    return Rate_ab, Rate_em


def electron_impurity_transfer_energy(dist_2D, Rate, stept):
    tempEnergy = dist_2D[:, 5].clip(0.001)
    Num = len(dist_2D)
    P = stept * Rate  # scattering probability
    random_P = np.random.uniform(0, 1, Num)
    P_ind = random_P <= P
    energy_ind = dist_2D[:, 5] >= 0
    happen = P_ind.astype(int)
    E_loss = np.random.uniform(
        0, tempEnergy - E_T) * happen * energy_ind
    dist_2D[:, 5] = dist_2D[:, 5] - E_loss
    return dist_2D, happen


def electron_hole_transfer_energy(dist_2D, hole_energy, Rate, stept):
    tempEnergy = dist_2D[:, 5].clip(0)
    Num = len(dist_2D)
    P_eh = stept * Rate
    random_P_eh = np.random.uniform(0, 1, Num)
    P_eh_ind = random_P_eh <= P_eh
    energy_eh_ind = dist_2D[:, 5] >= 0
    happen = P_eh_ind.astype(int)
    min_h = np.mean(hole_energy)
    eh_loss = np.random.uniform(-min_h, tempEnergy - min_h) * \
        happen * energy_eh_ind
    dist_2D[:, 5] = dist_2D[:, 5] - eh_loss
    hole_energy = hole_energy + np.mean(eh_loss)
    hole_energy = np.abs(hole_energy)
    return dist_2D, hole_energy, happen


def electron_acoustic_tranfer_energy(dist_2D, Rate, stept):
    Num = len(dist_2D)
    P_ac = stept * Rate
    # energy transfer
    ac_energy = 2 * m_T * dist_2D[:, 4] * ul *\
        np.cos(np.random.uniform(0, two_pi, Num))
    # scatterred electron index
    P_ac_ind = np.random.uniform(0, 1, Num) <= P_ac
    happen = P_ac_ind.astype(int)
    dist_2D[:, 5] = dist_2D[:, 5] - ac_energy * happen
    return dist_2D, happen


def electron_optical_transfer_energy(dist_2D, new_valley_dist, Rate_ab,
                                     Rate_em, E_ab, E_em, stept):
    new_valley_dist = new_valley_dist.tolist()
    Num = len(dist_2D)
    P_ab = stept * Rate_ab
    P_ab_ind = np.random.uniform(0, 1, Num) <= P_ab
    happen_ab = P_ab_ind.astype(int)
    dist_2D[:, 5] = dist_2D[:, 5] + E_ab * happen_ab
    new = dist_2D[P_ab_ind, :]
    dist_2D = dist_2D[(~P_ab_ind), :]
    new_valley_dist.extend(new.tolist())

    Num = len(dist_2D)
    Rate_em = Rate_em[(~P_ab_ind)]
    P_em = stept * Rate_em
    P_em_ind = np.random.uniform(0, 1, Num) <= P_em
    happen_em = P_em_ind.astype(int)
    dist_2D[:, 5] = dist_2D[:, 5] - E_em * happen_em
    new = dist_2D[P_em_ind, :]
    dist_2D = dist_2D[(~P_em_ind), :]
    new_valley_dist.extend(new.tolist())
    new_valley_dist = np.array(new_valley_dist)
    new_valley_dist = renew_distribution(new_valley_dist, 1)
    return dist_2D, new_valley_dist


def renew_distribution(dist_2D, happen):
    # ---- renew the velocity and direction after scattering -----
    Num = len(dist_2D)
    energy_ind = dist_2D[:, 5] > 0
    happen = happen * energy_ind
    phi = two_pi * np.random.uniform(0, 1, Num)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1, Num))
    dist_2D[:, 4] = dist_2D[:, 4] * (~energy_ind) + np.sqrt(
        2 * np.abs(dist_2D[:, 5]) * ec / m_T) * 10**9 * happen
    dist_2D[:, 2] = dist_2D[:, 4] * np.cos(theta) * happen + \
        dist_2D[:, 2] * (~energy_ind)
    dist_2D[:, 3] = dist_2D[:, 4] * np.sin(theta) * np.cos(phi) +\
        dist_2D[:, 3] * (~energy_ind)
    return dist_2D


def electron_transport(distribution_2D, types):
    '''electron transport in the conduction banc (CB) and suffer scattering:
    1. e-phonon (e-p) scattering:
        gain or loss the energy of a phonon (ep) after each scattering
    2. e-e scattering:
        a. scattering with electrons in CB can be ignored
        b. scattering with electrons in VB when energy bigger than Eg
    3. e-h scattering:
        main scattering mechanism in p-type GaAs, loss most energy
    4. e-impurity scattering:
        non-charged scattering can be ignored
        charged scattering is considered here
    '''
    surface_2D = []
    trap_2D = []
    back_2D = []
    time_data = []
    if types == 1:
        '''
        including e-ph, e-h and e-impurity scattering
        only for Gamma valley
        '''
        dist_2D = distribution_2D
        t = 0
        # assuming holes are steady state of maxwell-boltzmann distribution
        hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), len(dist_2D))
        hole_energy = m_h * hole_velocity**2 / 2 / ec
        time_data.append([t * 10**12, np.mean(dist_2D[:, 5]) * 10**3, 0,
                          len(dist_2D)])
        while t < total_time:
            tempEnergy = dist_2D[:, 5].clip(0.001)
            Num = len(dist_2D)
            # e-impurity scattering rate, (Y. Nishimura, Jnp. J. Appl. Phys.)
            Rate_ei = impurity_scattering(dist_2D[:, 5], 1)
            # e-h scattering rate
            Rate_eh = electron_hole_scattering(dist_2D[:, 5], 1)
            # acounstic phonon scattering rate
            Rate_ac = acoustic_phonon_scattering(dist_2D[:, 5], 1)

            Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(
                dist_2D[:, 5], 1)
            Rate_TX_ab, Rate_TX_em = optical_phonon_scattering(
                dist_2D[:, 5], 2)

            Rate = np.max([np.mean(Rate_ei), np.mean(Rate_eh),
                           np.mean(Rate_ac), np.mean(Rate_TL_ab),
                           np.mean(Rate_TL_em), np.mean(Rate_TX_ab),
                           np.mean(Rate_TX_em)])
            stept = 1 / 2 / Rate
            t += stept
            # transfer matrix after stept for electron without scattering
            M_st = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                             [stept, 0, 1, 0, 0, 0], [0, stept, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
            dist_2D = np.dot(dist_2D, M_st)

            # ----------- scattering change the distribution -----------
            # ----- get the energy distribution after scattering -------

            # ----- e-phonon scattering -----
            # -------- 1. acoustic phonon scattering -------
            # acoustic phonon scattering probability within stept
            P_ac = stept * Rate_ac
            # energy transfer
            ac_energy = 2 * m_T * dist_2D[:, 4] * ul *\
                np.cos(np.random.uniform(0, two_pi, Num))
            # scatterred electron index
            P_ac_ind = np.random.uniform(0, 1, Num) <= P_ac
            happen_ac = P_ac_ind.astype(int)
            dist_2D[:, 5] = dist_2D[:, 5] - ac_energy * happen_ac

            # -------- 2. optical phonon scattering from Gamma to L valley ----
            P_TL_ab = stept * Rate_TL_ab
            P_TL_ab_ind = np.random.uniform(0, 1, Num) <= P_TL_ab
            happen_TL_ab = P_TL_ab_ind.astype(int)
            dist_2D[:, 5] = dist_2D[:, 5] + (phonon_T_L - E_L_T) * happen_TL_ab

            P_TL_em = stept * Rate_TL_em
            P_TL_em_ind = np.random.uniform(0, 1, Num) <= P_TL_em
            happen_TL_em = P_TL_em_ind.astype(int)
            dist_2D[:, 5] = dist_2D[:, 5] - (phonon_T_L + E_L_T) * happen_TL_em

            # -------- 3. optical phonon scattering from Gamma to X valley ----
            P_TX_ab = stept * Rate_TX_ab
            P_TX_ab_ind = np.random.uniform(0, 1, Num) <= P_TX_ab
            happen_TX_ab = P_TX_ab_ind.astype(int)
            dist_2D[:, 5] = dist_2D[:, 5] + (phonon_T_X - E_X_T) * happen_TX_ab

            P_TX_em = stept * Rate_TX_em
            P_TX_em_ind = np.random.uniform(0, 1, Num) <= P_TX_em
            happen_TX_em = P_TX_em_ind.astype(int)
            dist_2D[:, 5] = dist_2D[:, 5] - (phonon_T_X + E_X_T) * happen_TX_em

            # ----- e-impurity scattering ---
            P_ei = stept * Rate_ei  # e-impurity scattering probability
            # print(tempEnergy[0], P_ei[0])
            random_P_ei = np.random.uniform(0, 1, Num)
            P_ei_ind = random_P_ei <= P_ei
            energy_ei_ind = dist_2D[:, 5] >= 0
            happen_ie = P_ei_ind.astype(int)
            ei_loss = np.random.uniform(
                0, tempEnergy - E_T) * happen_ie * energy_ei_ind
            dist_2D[:, 5] = dist_2D[:, 5] - ei_loss

            # ----- e-h scattering -----
            P_eh = stept * Rate_eh
            random_P_eh = np.random.uniform(0, 1, Num)
            P_eh_ind = random_P_eh <= P_eh
            energy_eh_ind = dist_2D[:, 5] >= 0
            happen_eh = P_eh_ind.astype(int)
            min_h = np.mean(hole_energy)
            eh_loss = np.random.uniform(-min_h, tempEnergy - min_h) * \
                happen_eh * energy_eh_ind
            dist_2D[:, 5] = dist_2D[:, 5] - eh_loss
            hole_energy = hole_energy + np.mean(eh_loss)
            hole_energy = np.abs(hole_energy)
            # print(dist_2D[:, 5], len(dist_2D))
            # print(np.mean(hole_energy), np.mean(dist_2D[:, 5]))

            happen = happen_ac + happen_ie + happen_eh

            # ---- renew the velocity and direction after scattering -----
            energy_ind = dist_2D[:, 5] > 0
            happen = happen * energy_ind
            r = np.random.uniform(0, 1, Num)
            phi = two_pi * r
            theta = np.arccos(1 - 2 * r)
            dist_2D[:, 4] = dist_2D[:, 4] * (~energy_ind) + np.sqrt(
                2 * np.abs(dist_2D[:, 5]) * ec / m_T) * 10**9 * happen
            dist_2D[:, 2] = dist_2D[:, 4] * np.cos(theta) * happen + \
                dist_2D[:, 2] * (~energy_ind)
            dist_2D[:, 3] = dist_2D[:, 4] * np.sin(theta) * np.cos(phi) +\
                dist_2D[:, 3] * (~energy_ind)

            # ------ filter electrons-------
            bd, fd, td, dist_2D = filter(dist_2D, 0.0001)

            back_2D.extend(bd.tolist())
            trap_2D.extend(td.tolist())
            surface_2D.extend(fd.tolist())

            time_data.append([t * 10**12, np.mean(dist_2D[:, 5]) * 10**3,
                              len(surface_2D), len(dist_2D)])

            if len(dist_2D) == 0:
                break
        '''
        fig, ax = plt.subplots()
        ax.hist(dist_2D[:, 5], bins=100)
        plt.show()'''

        dist_2D = dist_2D.tolist()
        dist_2D.extend(back_2D)
        dist_2D.extend(trap_2D)
        dist_2D.extend(surface_2D)
        dist_2D = np.array(dist_2D)
        dist_2D[:, 5] = np.maximum(dist_2D[:, 5], 0)
        dist_2D[:, 0] = np.clip(dist_2D[:, 0], surface, thick)

    elif types == 2:
        '''
        including e-ph, e-h and e-impurity scattering
        including Gamma, L and X valley scattering
        Assume all photoexcited electrons are in Gamma valley
        '''
        dist_2D = distribution_2D  # electrons in Gamma valley
        dist_L = np.array([])  # no electrons in L valley before scattering
        dist_X = np.array([])  # no electrons in X valley before scattering
        t = 0
        # assuming holes are steady state of maxwell-boltzmann distribution
        # hole temperature T_h = T = 298 K
        hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), len(dist_2D))
        hole_energy = m_h * hole_velocity**2 / 2 / ec
        time_data.append([t * 10**12, np.mean(dist_2D[:, 5]) * 10**3, 0,
                          len(dist_2D), 0, 0])
        while t < total_time:
            # electrons in Gamma valley begin scattering
            tempEnergy = dist_2D[:, 5].clip(0.001)
            Num = len(dist_2D)
            # e-impurity scattering rate, (Y. Nishimura, Jnp. J. Appl. Phys.)
            Rate_ei = impurity_scattering(dist_2D[:, 5], 1)
            # e-h scattering rate
            Rate_eh = electron_hole_scattering(dist_2D[:, 5], 1)
            # acounstic phonon scattering rate
            Rate_ac = acoustic_phonon_scattering(dist_2D[:, 5], 1)

            Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(
                dist_2D[:, 5], 1)
            Rate_TX_ab, Rate_TX_em = optical_phonon_scattering(
                dist_2D[:, 5], 2)

            Rate = np.max([np.mean(Rate_ei), np.mean(Rate_eh),
                           np.mean(Rate_ac), np.mean(Rate_TL_ab),
                           np.mean(Rate_TL_em), np.mean(Rate_TX_ab),
                           np.mean(Rate_TX_em)])
            stept = 1 / Rate / 5
            t += stept
            # transfer matrix after stept for electron without scattering
            M_st = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                             [stept, 0, 1, 0, 0, 0], [0, stept, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
            dist_2D = np.dot(dist_2D, M_st)

            # ----------- scattering change the distribution -----------
            # ----- get the energy distribution after scattering -------

            # --------- A. scattering for electrons in Gamma valley --------
            # ----- e-impurity scattering ---
            Rate_ei = impurity_scattering(dist_2D[:, 5], 1)
            dist_2D, happen_ie = electron_impurity_transfer_energy(
                dist_2D, Rate_ei, stept)
            dist_2D = renew_distribution(dist_2D, happen_ie)

            # ----- e-h scattering -----
            Rate_eh = electron_hole_scattering(dist_2D[:, 5], 1)
            dist_2D, hole_energy, happen_eh = electron_hole_transfer_energy(
                dist_2D, hole_energy, Rate_eh, stept)
            dist_2D = renew_distribution(dist_2D, happen_eh)

            # ----- e-phonon scattering -----
            # -------- 1. acoustic phonon scattering -------
            # acoustic phonon scattering probability within stept
            Rate_ac = acoustic_phonon_scattering(dist_2D[:, 5], 1)
            dist_2D, happen_ac = electron_acoustic_tranfer_energy(
                dist_2D, Rate_ac, stept)
            dist_2D = renew_distribution(dist_2D, happen_ac)

            # -------- 2. optical phonon scattering from Gamma to L valley ----
            Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(dist_2D[
                                                               :, 5], 1)
            E_TL_ab = E_L_T - phonon_T_L
            E_TL_em = E_L_T + phonon_T_L
            dist_2D, dist_L = electron_optical_transfer_energy(
                dist_2D, dist_L, Rate_TL_ab, Rate_TL_em, E_TL_ab, E_TL_em,
                stept)

            # -------- 3. optical phonon scattering from Gamma to X valley ----
            Rate_TX_ab, Rate_TX_em = optical_phonon_scattering(dist_2D[
                                                               :, 5], 2)
            E_TX_ab = E_X_T - phonon_T_X
            E_TX_em = E_X_T + phonon_T_X
            dist_2D, dist_X = electron_optical_transfer_energy(
                dist_2D, dist_X, Rate_TX_ab, Rate_TX_em, E_TX_ab, E_TX_em,
                stept)

            # ------- B. scattering for electrons in L valley -------
            # ----- e-impurity scattering ---
            Rate_ei = impurity_scattering(dist_L[:, 5], 2)
            dist_L, happen_ie = electron_impurity_transfer_energy(
                dist_L, Rate_ei, stept)
            dist_L = renew_distribution(dist_L, happen_ie)

            # ----- e-h scattering -----
            Rate_eh = electron_hole_scattering(dist_L[:, 5], 2)
            dist_L, hole_energy, happen_eh = electron_hole_transfer_energy(
                dist_L, hole_energy, Rate_eh, stept)
            dist_L = renew_distribution(dist_L, happen_eh)

            # ----- e-phonon scattering -----
            # -------- 1. acoustic phonon scattering -------
            # acoustic phonon scattering probability within stept
            Rate_ac = acoustic_phonon_scattering(dist_L[:, 5], 2)
            dist_L, happen_ac = electron_acoustic_tranfer_energy(
                dist_L, Rate_ac, stept)
            dist_L = renew_distribution(dist_L, happen_ac)

            # -------- 2. optical phonon scattering from L to Gamma valley ----
            Rate_LT_ab, Rate_LT_em = optical_phonon_scattering(dist_L[
                                                               :, 5], 3)
            E_LT_ab = -E_L_T - phonon_T_L
            E_LT_em = -E_L_T + phonon_T_L
            dist_L, dist_2D = electron_optical_transfer_energy(
                dist_L, dist_2D, Rate_LT_ab, Rate_LT_em, E_LT_ab, E_LT_em,
                stept)

            # -------- 3. optical phonon scattering from L to L valley ----
            Rate_LL_ab, Rate_LL_em = optical_phonon_scattering(dist_L[
                                                               :, 5], 4)
            E_LL_ab = -phonon_L_L
            E_LL_em = phonon_L_L
            dist_L, dist_L = electron_optical_transfer_energy(
                dist_L, dist_L, Rate_LL_ab, Rate_LL_em, E_LL_ab, E_LL_em,
                stept)

            # -------- 4. optical phonon scattering from L to X valley ----
            Rate_LX_ab, Rate_LX_em = optical_phonon_scattering(dist_L[
                                                               :, 5], 4)
            E_LX_ab = E_X_T - E_L_T - phonon_L_X
            E_LX_em = E_X_T - E_L_T + phonon_L_X
            dist_L, dist_X = electron_optical_transfer_energy(
                dist_L, dist_X, Rate_LX_ab, Rate_LX_em, E_LX_ab, E_LX_em,
                stept)

            # -------- C. scattering for electrons in X valley -------

            # ------ filtting electrons-------
            # -------- filtting electrons in Gamma valley --------
            bd, fd, td, dist_2D = filter(dist_2D, 0)
            back_2D.extend(bd.tolist())
            trap_2D.extend(td.tolist())
            surface_2D.extend(fd.tolist())

            # -------- filtting electrons in L valley --------
            bd, fd, td, dist_L = filter(dist_L, 0)
            back_2D.extend(bd.tolist())
            trap_2D.extend(td.tolist())
            surface_2D.extend(fd.tolist())

            # -------- filtting electrons in X valley --------
            bd, fd, td, dist_X = filter(dist_X, 0)
            back_2D.extend(bd.tolist())
            trap_2D.extend(td.tolist())
            surface_2D.extend(fd.tolist())

            time_data.append([t * 10**12, np.mean(dist_2D[:, 5]) * 10**3,
                              len(surface_2D), len(dist_2D), len(dist_L),
                              len(dist_X)])
            # print(len(dist_2D) + len(dist_L) + len(dist_X) + len(surface_2D))
            if (len(dist_2D) + len(dist_L) + len(dist_X)) == 0:
                break
        '''
        fig, ax = plt.subplots()
        ax.hist(dist_2D[:, 5], bins=100)
        plt.show()'''

        dist_2D = dist_2D.tolist()
        dist_2D.extend(back_2D)
        dist_2D.extend(trap_2D)
        dist_2D.extend(surface_2D)
        dist_2D = np.array(dist_2D)
        dist_2D[:, 5] = np.maximum(dist_2D[:, 5], 0)
        dist_2D[:, 0] = np.clip(dist_2D[:, 0], surface, thick)

    else:
        print('Wrong electron transport types')

    time_data = np.array(time_data)

    back_2D = np.array(back_2D)
    if len(back_2D) != 0:
        back_2D[:, 0] = thick

    trap_2D = np.array(trap_2D)
    if len(trap_2D) != 0:
        trap_2D[:, 5] = 0

    surface_2D = np.array(surface_2D)
    if len(surface_2D) != 0:
        surface_2D[:, 0] = surface

    return surface_2D, back_2D, trap_2D, dist_2D, time_data


def filter(dist_2D, threshold_value):
    ''' filter electrons
    Find these electrons diffused to surface, substrate, trapped
    and the rest electrons that will continue to diffuse
    '''
    assert thick > surface
    back = dist_2D[:, 0] >= thick
    front = dist_2D[:, 0] <= surface
    trap = dist_2D[:, 5] <= threshold_value

    back_dist = dist_2D[back, :]
    front_dist = dist_2D[front, :]
    trap_dist = dist_2D[trap, :]
    rest_dist = dist_2D[(~back) & (~front) & (~trap), :]
    return back_dist, front_dist, trap_dist, rest_dist


def electron_emitting(surface_2D):
    ''' two conidtion should be matched before emitting:
    1. E_out = E_e - E_A + E_sch > 0
    2. P_out > P_out_T = P_in_T
    '''
    surface_trap = []
    match_ind = surface_2D[:, 5] >= (E_A - E_sch)
    match_E = surface_2D[match_ind, :]
    surface_trap.extend(surface_2D[(~match_ind), :].tolist())
    match_E = match_E - E_A - E_sch

    phi = np.random.uniform(0, 2 * pi, (len(match_E), 1))
    match_E = np.append(match_E, phi, 1)
    match_E[:, 4] = np.sqrt(2 * match_E[:, 5] / m_T) * \
        c * 10**9 * np.cos(match_E[:, 6])
    match = np.abs(match_E[:, 4]) >= np.abs(match_E[:, 3])
    emission_2D = match_E[match, :]
    emission_2D[:, 2] = np.sqrt(emission_2D[:, 4]**2 - emission_2D[:, 3]**2)
    surface_trap.extend(match_E[(~match), :].tolist())
    surface_trap = np.array(surface_trap)
    return emission_2D, surface_trap


def plot_QE(filename, data):
    exp_data = np.genfromtxt('GaAs_QE_experiment.csv', delimiter=',')
    exp_data[:, 0] = 1240 / exp_data[:, 0]
    fig1, ax1 = plt.subplots()
    ax1.plot(data[:, 0], data[:, 1], exp_data[:, 0], exp_data[:, 1], '.')
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=14)
    ax1.set_ylabel(r'QE (%)', fontsize=14)
    ax1.tick_params('both', direction='in', labelsize=12)
    plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def plot_time_data(time_data):
    fig1, ax1 = plt.subplots()
    ax1.plot(time_data[:, 0], time_data[:, 1], 'b')
    ax1.set_xlabel('Time (ps)', fontsize=14)
    ax1.set_ylabel('Energy (meV)', fontsize=14, color='b')
    ax1.tick_params('y', color='b')
    ax1.tick_params('both', direction='in', labelsize=12)

    ax2 = ax1.twinx()
    ax2.semilogy(time_data[:, 0], time_data[:, 2],
                 time_data[:, 0], time_data[:, 3],
                 time_data[:, 0], time_data[:, 4],
                 time_data[:, 0], time_data[:, 5])
    ax2.set_ylabel('Counts', fontsize=14)
    ax2.tick_params('y', color='r')
    ax1.tick_params('both', direction='in', labelsize=12)
    ax2.legend(['Surface', 'Gamma', 'L', 'X'],
               loc='center', frameon=False, fontsize=12)
    fig1.tight_layout()
    plt.show()


def save_date(filename, data):
    types = 1
    # type 1 control the number of saved data
    # type 2 save all data
    if types == 0:
        file = open(filename, 'w')
        for i in range(len(data)):
            for j in range(np.size(data[0, :]) - 1):
                file.write('%.2f' % data[i, j])
                file.write(',')
            file.write('%.2f' % data[i, -1])
            file.write('\n')
        file.close()
    elif types == 1:
        np.savetxt(filename + '.csv', data, delimiter=',', fmt='%.2f')


def plot_scattering_rate(types):
    '''
    types=1, plot the scattering rate for the electrons in Gamma valley
    types=2, plot the scattering rate for the electrons in L valley
    types=3, plot the scattering rate for the electrons in X valley
    '''
    if types == 1:
        # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Ni)
        # hole_energy = m_h * hole_velocity**2 / 2 / ec
        e_energy = np.linspace(0, 1.5, Ni)
        # hole_energy = np.linspace(0, 0.1, Ni)
        # carrier_scattering(e_energy, hole_energy)
        Rate_eh = electron_hole_scattering(e_energy, 1)
        Rate_ei = impurity_scattering(e_energy, 1)
        Rate_ac = acoustic_phonon_scattering(e_energy, 1)
        Rate_TL_ab, Rate_TL_em = optical_phonon_scattering(
            e_energy, 1)
        Rate_TX_ab, Rate_TX_em = optical_phonon_scattering(
            e_energy, 2)
        fig, ax = plt.subplots()
        ax.semilogy(e_energy, Rate_eh, '.', e_energy, Rate_ei, '.',
                    e_energy, Rate_ac, '.', e_energy, Rate_TL_ab, '.',
                    e_energy, Rate_TL_em, '.', e_energy, Rate_TX_ab, '.',
                    e_energy, Rate_TX_em, '.')
        ax.set_xlabel(r'Electron energy (eV)', fontsize=16)
        ax.set_ylabel(r'scattering rate ($s^{-1}$)', fontsize=16)
        plt.legend(['e-hole', 'e-impurity', 'acoustic phonon', 'T to L absorb',
                    'T to L emission', 'T to X absorb', 'T to X emission'],
                   frameon=False, fontsize=12)
        plt.tight_layout()
        plt.show()
    elif types == 2:
        # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Ni)
        # hole_energy = m_h * hole_velocity**2 / 2 / ec
        e_energy = np.linspace(0, 1.5, Ni)
        # hole_energy = np.linspace(0, 0.1, Ni)
        Rate_eh = electron_hole_scattering(e_energy, 2)
        Rate_ei = impurity_scattering(e_energy, 2)
        Rate_ac = acoustic_phonon_scattering(e_energy, 2)
        Rate_LT_ab, Rate_LT_em = optical_phonon_scattering(
            e_energy, 3)
        Rate_LL_ab, Rate_LL_em = optical_phonon_scattering(
            e_energy, 4)
        Rate_LX_ab, Rate_LX_em = optical_phonon_scattering(
            e_energy, 5)
        fig, ax = plt.subplots()
        ax.semilogy(e_energy, Rate_eh, '.', e_energy, Rate_ei, '.',
                    e_energy, Rate_ac, '.', e_energy, Rate_LT_ab, '.',
                    e_energy, Rate_LT_em, '.', e_energy, Rate_LL_ab, '.',
                    e_energy, Rate_LL_em, '.', e_energy, Rate_LX_ab, '.',
                    e_energy, Rate_LX_em, '.')
        ax.set_xlabel(r'Electron energy (eV)', fontsize=16)
        ax.set_ylabel(r'scattering rate ($s^{-1}$)', fontsize=16)
        ax.set_xlim([0, 1.5])
        plt.legend(['e-hole', 'e-impurity', 'acoustic phonon', 'L to T absorb',
                    'L to T emission', 'L to L absorb', 'L to L emission',
                    'L to X absorb', 'L to X emission'],
                   frameon=False, fontsize=12)
        plt.tight_layout()
        plt.show()
    elif types == 3:
        # hole_velocity = maxwell.rvs(0, np.sqrt(kB * T / m_h), Ni)
        # hole_energy = m_h * hole_velocity**2 / 2 / ec
        e_energy = np.linspace(0, 1.5, Ni)
        # hole_energy = np.linspace(0, 0.1, Ni)
        Rate_eh = electron_hole_scattering(e_energy, 3)
        Rate_ei = impurity_scattering(e_energy, 3)
        Rate_ac = acoustic_phonon_scattering(e_energy, 3)
        Rate_XT_ab, Rate_XT_em = optical_phonon_scattering(
            e_energy, 6)
        Rate_XL_ab, Rate_XL_em = optical_phonon_scattering(
            e_energy, 7)
        Rate_XX_ab, Rate_XX_em = optical_phonon_scattering(
            e_energy, 8)
        fig, ax = plt.subplots()
        ax.semilogy(e_energy, Rate_eh, '.', e_energy, Rate_ei, '.',
                    e_energy, Rate_ac, '.', e_energy, Rate_XT_ab, '.',
                    e_energy, Rate_XT_em, '.', e_energy, Rate_XL_ab, '.',
                    e_energy, Rate_XL_em, '.', e_energy, Rate_XX_ab, '.',
                    e_energy, Rate_XX_em, '.')
        ax.set_xlabel(r'Electron energy (eV)', fontsize=16)
        ax.set_ylabel(r'scattering rate ($s^{-1}$)', fontsize=16)
        ax.set_xlim([0, 1.5])
        plt.legend(['e-hole', 'e-impurity', 'acoustic phonon', 'X to T absorb',
                    'X to T emission', 'X to L absorb', 'X to L emission',
                    'X to X absorb', 'X to X emission'],
                   frameon=False, fontsize=12)
        plt.tight_layout()
        plt.show()


def main(opt):
    hw_start = float('%.2f' % Eg) + 0.01  # eV
    hw_end = 2.48  # eV
    hw_step = 0.02  # eV
    hw_test = 2.0  # eV
    data = []
    filename = 'QE_' + str(thick) + '_' + str(E_A)
    if opt == 1:  # for test
        dist_2D = electron_distribution(hw_test, 2)
        print('excited electron ratio: ', len(dist_2D) / Ni)

        surface_2D, back_2D, trap_2D, dist_2D, time_data = \
            electron_transport(dist_2D, 2)
        print('surface electron ratio: ', len(surface_2D) / Ni)

        emiss_2D, surf_trap = electron_emitting(surface_2D)
        print('QE (%): ', 100.0 * len(emiss_2D) / Ni)

        plot_time_data(time_data)

    elif opt == 2:
        for hw in np.arange(hw_start, hw_end, hw_step):
            dist_2D = electron_distribution(hw, 2)
            print('excited electron ratio: ', len(dist_2D) / Ni)

            surface_2D, back_2D, trap_2D, dist_2D, energy_time = \
                electron_transport(dist_2D, 2)
            print('surface electron ratio: ', len(surface_2D) / Ni)

            emiss_2D, surf_trap = electron_emitting(surface_2D)

            QE = 100.0 * len(emiss_2D) / Ni
            print('photon energy (eV): ', hw, ', QE (%): ', QE)
            data.append([hw, QE])

        data = np.array(data)
        save_date(filename, data)
        plot_QE(filename, data)
    elif opt == 3:
        plot_scattering_rate(2)
    else:
        print('Wrong run option')

    print('run time:', time.time() - start_time, 's')


if __name__ == '__main__':
    main(1)
