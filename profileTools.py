# Tools for generating planetesimal and gas profiles in a PPD

import numpy as np
import astropy.units as u
from astropy.constants import G, k_B, m_p

# Convert between semimajor axis and orbital period (cgs units)
def sma(t_orb, m_central):
    return (G.cgs.value*m_central*(t_orb/(2*np.pi))**2)**(1/3)
def t_orb(sma, m_central):
    return 2*np.pi*np.sqrt(sma**3/(G.cgs.value*m_central))

# Planetesimal surface density profile, r in AU
def sigma_pl(alpha, A, r, f):
    return f*A*r**(-alpha)

# Convert solid surface density to gas surface density
def sigma_gas(surf_den_pl, gas_to_dust):
    return surf_den_pl/gas_to_dust

# Midplane mass density of gas
def rho_gas(cs, omega, surf_den_pl, gas_to_dust):
    surf_den_gas = sigma_gas(surf_den_pl, gas_to_dust)
    h_gas = cs/omega
    return surf_den_gas/h_gas

# Temperature profile of disk, r in AU
def T_prof(T0, q, r):
    return T0*r**(-q)

# Sound speed profile of disk
def soundspeed(T0, mu, q, r_AU):
    T = T_prof(T0, q, r_AU)
    return np.sqrt(k_B.cgs.value*T/(mu*m_p.cgs.value))

# Toomre q parameter
def toom_q(cs, omega, sigma):
    return cs*omega/(np.pi*G.cgs*sigma)

# Headwind velocity of gas in pressure-supported disk
# r and m_central in cm and M_sun, cs in cm/s
# q is the power law index of the pressure profile, which should be same as
# temperature profile for an ideal gas
def v_gas(m_central, r, cs, q):
    m_central_cgs = (m_central*u.M_sun).to(u.g).value
    v_k = np.sqrt(G.cgs.value*m_central_cgs/r)
    return v_k*(1 - np.sqrt(1 - (q*cs**2/v_k**2)))

# Stopping time of planetesimal in stokes regime
def t_stop(m_pl, s_pl, C_D, rho_gas, v_gas):
    return 2*m_pl/(C_D*np.pi*s_pl**2*rho_gas*v_gas)

# Radial drift velocity of planetesimal in stokes regime (Weidenschilling 1997)
def vr_drift(m_pl, s_pl, C_D, rho_gas, v_gas, omega):
    ts = t_stop(m_pl, s_pl, C_D, rho_gas, v_gas)
    return 2*v_gas/(omega*ts + 1/(omega*ts))

# Relaxation time, equal to viscous stirring timescale in dispersion dominated regime
def t_relax(v, n, m):
    logLam = 3 # Is this true at short period?
    return v**3/(np.pi*G.cgs.value**2*n*m**2*logLam)

# Safronov collision timescale
def t_coll(r, n_pl, v_pl, mp, rp):
    vesc = np.sqrt(G.cgs.value*mp/rp)
    theta = vesc/v_pl
    cross = np.pi*(rp)**2*(1 + theta**2)
    return 1/(n_pl*cross*v_pl)

# Equilibirum eccentricity of planetesimals in gas
# Derived by equating relaxation time (Ida 1993)
# With the stopping time (Adachi 1976)
def e_eq(omega, sigma_pl, m_pl, s_pl, C_D, vk, rho_gas, v_gas):
    logLam = 3
    return (G.cgs.value**2*m_pl**2*logLam*omega*sigma_pl/(C_D*s_pl**2*vk**4*rho_gas*v_gas))**(1/4)
