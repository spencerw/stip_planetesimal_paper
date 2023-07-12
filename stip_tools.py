import numpy as np
import astropy.units as u
from astropy.constants import G
import KeplerOrbit as ko
import pandas as pd
import pynbody as pb

# Each node represents a particle
class Node:
    def __init__(self):
        self.iord = -1
        self.children = []
        self.parent = None
        self.parent_time = 99999999999999 # Time at which this node connected to the parent

# Load a changa output file as a pynbody snapshot
# deltaT - timestep in 2 pi years
# mCentral - central mass in Msun
def load_changa(filename, deltaT, mCentral):
    snap = pb.load(filename)
    pl = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
    if filename.split('.')[-1].isnumeric():
        time = int(filename.split('.')[-1])*deltaT/(2*np.pi)/1e6
    else:
        time = 0.0

    return pl, time

# Load a genga output file as a pynbody snapshot
# deltaT - timestep in days
# mCentral - central mass in Msun
def load_genga(filename, deltaT, mCentral):
    nam = ['t', 'i1', 'm1', 'r1', 'x1', 'y1', 'z1', 'vx1', 'vy1', 'vz1', 'Sx1', 'Sy1', 'Sz1', 'amin1', 'amax1', 'emin1', 'emax1', 'aecount1', 'aecountT1', 'enccountT1', 'test1']
    time = (int(filename.split('_')[-1].split('.')[0])*deltaT*u.d).to(u.Myr).value
    df = pd.read_csv(filename, names=nam, sep=' ', index_col=False)
    a, e, inc, asc_node, omega, M = ko.cart2kep(df['x1'], df['y1'], df['z1'], \
                                                df['vx1'], df['vy1'], df['vz1'], \
                                                mCentral, df['m1'])
    snap = pb.new(len(df))
    snap['pos'][:,0] = df['x1']
    snap['pos'][:,1] = df['y1']
    snap['pos'][:,2] = df['z1']
    snap['vel'][:,0] = df['vx1']
    snap['vel'][:,1] = df['vy1']
    snap['vel'][:,2] = df['vz1']
    snap['mass'] = df['m1']
    snap['rad'] = df['r1']
    snap['iord'] = df['i1']

    pl = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
    return pl, time

# Open a changa collision logfile and return a pandas table
def load_changa_coll(filename):
    nam = ['time', 'collType', 'iord1', 'iord2', 'm1', 'm2', 'r1', 'r2', 'x1x', 'x1y', 'x1z', 'x2x', 'x2y', 'x2z',  'v1x', 'v1y', 'v1z', 'v2x', 'v2y', 'v2z', 'w1x', 'w1y', 'w1z', 'w2x', 'w2y', 'w2z']
    coll = pd.read_csv(filename, names=nam, sep=' ', index_col=False)
    return coll

# Open a genga ejection logfile and return a pandas table
def load_genga_ej(filename):
    nam = ['time', 'index', 'm', 'r', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Sx', 'Sy', 'Sz', 'case']
    eject = pd.read_csv(filename, names=nam, sep=' ', index_col=False)
    return eject

# Open a genga collision logfile and return a pandas table
def load_genga_coll(filename):
    nam = ['time', 'indexi', 'mi', 'ri', 'xi', 'yi', 'zi', 'vxi', 'vyi', 'vzi', 'Sxi', \
       'Syi', 'Szi', 'indexj', 'mj', 'rj', 'xj', 'yj', 'zj', 'vxj', 'vyj', 'vzj', 'Sxj', 'Syj', 'Szj']
    coll = pd.read_csv(filename, names=nam, sep=' ', index_col=False)
    return coll

# Given a collision logfile from changa, determine which properties
# are associated with the deleted and surviving particle
def survivor_vec_changa(coll):
    # For ChaNGa, the more massive particle surives
    i1, i2 = coll['iord2'], coll['iord2']
    m1, m2 = coll['m1'], coll['m2']
    px1, py1, pz1 = coll['x1x'], coll['x1y'], coll['x1z']
    px2, py2, pz2 = coll['x2x'], coll['x2y'], coll['x2z']
    vx1, vy1, vz1 = coll['v1x'], coll['v1y'], coll['v1z']
    vx2, vy2, vz2 = coll['v2x'], coll['v2y'], coll['v2z']
    s_iord, del_iord = i1.copy().values, i2.copy().values
    s_mass, del_mass = m1.copy().values, m2.copy().values
    s_px, s_py, s_pz = px1.copy().values, py1.copy().values, pz1.copy().values
    del_px, del_py, del_pz = px2.copy().values, py2.copy().values, pz2.copy().values
    s_vx, s_vy, s_vz = vx1.copy().values, vy1.copy().values, vz1.copy().values
    del_vx, del_vy, del_vz = vx2.copy().values, vy2.copy().values, vz2.copy().values
    
    # More massive particle survives
    # If equal mass, lower index survives
    mass_mask = m1 > m2

    m = mass_mask
    s_iord[m], del_iord[m] = i1[m], i2[m]
    s_mass[m], del_mass[m] = m1[m], m2[m]
    s_px[m], s_py[m], s_pz[m], del_px[m], del_py[m], del_pz[m] =  px1[m], py1[m], pz1[m], px2[m], py2[m], pz2[m]
    s_vx[m], s_vy[m], s_vz[m], del_vx[m], del_vy[m], del_vz[m] =  vx1[m], vy1[m], vz1[m], vx2[m], vy2[m], vz2[m]

    time = coll['time'].values

    return {'s_iord': s_iord, 'del_iord': del_iord, 's_mass': s_mass, 'del_mass': del_mass, 's_px': s_px, 's_py': s_py,\
            's_pz': s_pz, 'del_px': del_px, 'del_py': del_py, 'del_pz': del_pz, 's_vx': s_vx, 's_vy': s_vy, \
            's_vz': s_vz, 'del_vx': del_vx, 'del_vy': del_vy, 'del_vz': del_vz, 'time': time}

# Given a collision logfile from genga, determine which properties
# are associated with the deleted and surviving particle
def survivor_vec_genga(coll):
    i1, i2 = coll['indexi'], coll['indexj']
    m1, m2 = coll['mi'], coll['mj']
    px1, py1, pz1 = coll['xi'], coll['yi'], coll['zi']
    px2, py2, pz2 = coll['xj'], coll['yj'], coll['zj']
    vx1, vy1, vz1 = coll['vxi'], coll['vyi'], coll['vzi']
    vx2, vy2, vz2 = coll['vxj'], coll['vyj'], coll['vzj']
    s_iord, del_iord = i1.copy().values, i2.copy().values
    s_mass, del_mass = m1.copy().values, m2.copy().values
    s_px, s_py, s_pz = px1.copy().values, py1.copy().values, pz1.copy().values
    del_px, del_py, del_pz = px2.copy().values, py2.copy().values, pz2.copy().values
    s_vx, s_vy, s_vz = vx1.copy().values, vy1.copy().values, vz1.copy().values
    del_vx, del_vy, del_vz = vx2.copy().values, vy2.copy().values, vz2.copy().values
    
    # More massive particle survives
    # If equal mass, lower index survives
    mass_mask = m1 > m2

    m = mass_mask
    s_iord[m], del_iord[m] = i1[m], i2[m]
    s_mass[m], del_mass[m] = m1[m], m2[m]
    s_px[m], s_py[m], s_pz[m], del_px[m], del_py[m], del_pz[m] =  px1[m], py1[m], pz1[m], px2[m], py2[m], pz2[m]
    s_vx[m], s_vy[m], s_vz[m], del_vx[m], del_vy[m], del_vz[m] =  vx1[m], vy1[m], vz1[m], vx2[m], vy2[m], vz2[m]

    equal_mass_mask = np.logical_and(m1 == m2, i1 < i2)
    m = equal_mass_mask
    s_iord[m], del_iord[m] = i1[m], i2[m]
    s_mass[m], del_mass[m] = m1[m], m2[m]
    s_px[m], s_py[m], s_pz[m], del_px[m], del_py[m], del_pz[m] =  px1[m], py1[m], pz1[m], px2[m], py2[m], pz2[m]
    s_vx[m], s_vy[m], s_vz[m], del_vx[m], del_vy[m], del_vz[m] =  vx1[m], vy1[m], vz1[m], vx2[m], vy2[m], vz2[m]

    equal_mass_mask = np.logical_and(m1 == m2, i1 > i2)
    m = equal_mass_mask
    s_iord[m], del_iord[m] = i2[m], i1[m]
    s_mass[m], del_mass[m] = m2[m], m1[m]
    s_px[m], s_py[m], s_pz[m], del_px[m], del_py[m], del_pz[m] =  px2[m], py2[m], pz2[m], px1[m], py1[m], pz1[m]
    s_vx[m], s_vy[m], s_vz[m], del_vx[m], del_vy[m], del_vz[m] =  vx2[m], vy2[m], vz2[m], vx1[m], vy1[m], vz1[m]

    time = coll['time'].values

    return {'s_iord': s_iord, 'del_iord': del_iord, 's_mass': s_mass, 'del_mass': del_mass, 's_px': s_px, 's_py': s_py,\
            's_pz': s_pz, 'del_px': del_px, 'del_py': del_py, 'del_pz': del_pz, 's_vx': s_vx, 's_vy': s_vy, \
            's_vz': s_vz, 'del_vx': del_vx, 'del_vy': del_vy, 'del_vz': del_vz, 'time': time}

def prev_iorder(new_iord, num_original, del_iord):
    indices = np.zeros(num_original, dtype=bool)
    indices[del_iord] = True
    prev_iord = np.where(indices == False)[0][new_iord]
    return prev_iord

# Convert semimajor axis to orbital period (in days)
# sma - semimajor axis in AU
# mCentral - central mass in Msun
# inDays - return per in days, otherwise sim units (2pi years)
def sma2per(sma, mCentral, inDays=True):
    if inDays:
        return (2*np.pi*np.sqrt(sma**3/mCentral)*u.yr).to(u.d).value/(2*np.pi)
    else:
        return 2*np.pi*np.sqrt(sma**3/mCentral)

# Convert orbital period to semimajor axis (in AU)
# per - orbital period in days
# mCentral - central mass in Msun
# perInDays - per given in days, otherwise 2pi years assumed
def per2sma(per, mCentral, perInDays=True):
    if perInDays:
        per_sim = (per*u.d).to(u.yr).value*(2*np.pi)
        return (mCentral*(per_sim/(2*np.pi))**2)**(1./3.)
    else:
        return (mCentral*(per/(2*np.pi))**2)**(1./3.)

# Generate a profile for the isolation mass as a function of radial distance
# using a pynbody snapshot. Return radial bins in AU and isolation mass in Mearth
# btilde - Assumed feeding zone size in mutual hill radii
# mCentral - central mass in Msun
# amin, amax - Extent of the radial profile in AU
def isoprof(snap, btilde, mCentral, amin=0.01, amax=0.2):
    a_vals = np.linspace(amin, amax)
    prof = pb.analysis.profile.Profile(snap, bins=a_vals)

    mCentralg = (mCentral*u.M_sun).to(u.g).value
    surf_den = (prof['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2).value
    m_iso = np.sqrt((2*np.pi*(prof['rbins']*u.AU).to(u.cm).value**2*btilde*surf_den)**3/(3*mCentralg))
    m_iso = (m_iso*u.g).to(u.M_earth).value

    return prof['rbins'], m_iso


