#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pynbody as pb
import natsort as ns
import glob as gl
import os
import profileTools as pt

from astropy import units as u
from astropy.constants import G
simT = u.year/(2*np.pi)
simV = u.AU / simT

from scipy import stats, optimize

import sys
sys.path.insert(0, '../OrbitTools/')
import OrbitTools

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

simT = u.year/(2*np.pi)
simV = u.AU/simT

clobber = True
fmt = 'png'
s = 0.005

from scipy.integrate import quad

def lnLam(b1, b2):
    return quad(lambda x: 1/x, b1, b2)[0]

sigma1 = 6
f_disk = 1
h = 1
alpha = 1

# Values for M stars from Backus + Quinn 2016
q = 0.59
T0 = 130
mu = 2.34 # Hayashi 1981

C_D = 1
dust_to_gas = [0.01, 0.1, 0.5]

rhop = 2
m_central = (0.13*u.M_sun).to(u.g).value
Rp = (50*u.km).to(u.cm).value
mp = 4/3*np.pi*Rp**3*rhop
f_pl = [1., 3., 6.]

def e_i(e_h):
    return e_h*(mp/(3*m_central))**(1/3)

def e_h(e):
    return e*(mp/(3*m_central))**(-1/3)

a_vals_au = np.logspace(-2, 0.5)
e_h_vals = np.logspace(0, 2)

a_vals = (a_vals_au*u.AU).to(u.cm).value
t_orbit = pt.t_orb(a_vals, m_central)
t_orbit_d = (t_orbit*u.s).to(u.d).value

sigma_pl = pt.sigma_pl(alpha, sigma1, a_vals_au, f_disk)

omega = np.sqrt(G.cgs.value*m_central/a_vals**3)
vk = np.sqrt(G.cgs.value*m_central/a_vals)

def plot_trel_tcoll_eq_gf():
	file_str = 'figures/time_eq_gf.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	t_relax_vals = np.empty((len(a_vals_au), len(e_h_vals)))
	t_coll_vals = np.empty((len(a_vals_au), len(e_h_vals)))
	t_eq_e_vals = np.empty((len(f_pl), len(a_vals_au)))

	for f_pl_idx in range(len(f_pl)):
	    for idx in range(len(a_vals)):
	        for idx1 in range(len(e_h_vals)):
	            e_std = e_i(e_h_vals[idx1])
	            i_std = e_std/2
	            omega = np.sqrt(G.cgs.value*m_central/a_vals[idx]**3)
	            vk = np.sqrt(G.cgs.value*m_central/a_vals[idx])
	            sigma = np.sqrt(e_std**2 + i_std**2)*vk
	            n = sigma_pl[idx]*omega/(2*mp*sigma)

	            bmin = 2*G.cgs.value*mp/sigma**2
	            bmax = a_vals[idx]
	            lnLamVal = lnLam(bmin, bmax)
	            t_relax = sigma**3/(n*np.pi*G.cgs.value**2*mp**2*lnLamVal)
	            t_relax_vals[idx][idx1] = t_relax

	            vesc = np.sqrt(G.cgs.value*mp/Rp)
	            theta = vesc/sigma
	            cross = np.pi*(Rp*f_pl[f_pl_idx])**2*(1 + theta**2)
	            t_coll_vals[idx][idx1] = 1/(n*cross*sigma)

	        first_ind = np.where(t_relax_vals[idx] > t_coll_vals[idx])[0]
	        if len(first_ind) > 0:
	            t_eq_e_vals[f_pl_idx][idx] = e_i(e_h_vals[first_ind][0])
	        else:
	            t_eq_e_vals[f_pl_idx][idx] = np.inf

	e_gf_vals = np.empty((len(f_pl), len(a_vals)))

	for f_pl_idx in range(len(f_pl)):
	    for idx in range(len(a_vals)):
	        e_gf_vals[f_pl_idx][idx] = np.sqrt(4*mp*a_vals[idx]/(3*m_central*(f_pl[f_pl_idx]*Rp)))

	fig, axes = plt.subplots(figsize=(8,8))
	for idx, val in enumerate(f_pl):
	    c = next(axes._get_lines.prop_cycler)['color']
	    axes.plot(t_orbit_d, t_eq_e_vals[idx], label=r'f$_{pl}$ = ' + str(int(val)), linestyle='-', color=c)
	    axes.plot(t_orbit_d, e_gf_vals[idx], linestyle='--', color=c)
	axes.legend()
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlabel(r'Orbital Period [d]')
	axes.set_ylabel(r'Equilibirum $\left< e^{2} \right>^{1/2}$')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_m_iso():
	file_str = 'figures/m_iso.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	def func(x, e_std, sigma_pl, a_val):
	    i_std = e_std/2
	    
	    etilde, itilde = e_std*(10**x/(3*m_central))**(-1/3), i_std*(10**x/(3*m_central))**(-1/3)
	    btilde = 2/np.sqrt(3)*np.sqrt(9 + etilde**2 + itilde**2)
	    
	    return np.log10(4*np.pi*a_val**2*btilde*sigma_pl*(10**x/(3*m_central))**(1/3)) - x

	m_iso_vals = np.empty((len(a_vals_au), len(e_h_vals)))
	for idx in range(len(a_vals)):
	    for idx1 in range(len(e_h_vals)):
	        m_iso_vals[idx][idx1] = 10**optimize.root(func, 20, args=(e_i(e_h_vals[idx1]), sigma_pl[idx], a_vals[idx])).x

	e_eq_vals = np.empty((len(dust_to_gas), len(t_orbit)))
	for idx in range(len(dust_to_gas)):
		cs = pt.soundspeed(T0, mu, q, a_vals_au)
		vgas = pt.v_gas((m_central*u.g).to(u.M_sun).value, a_vals, cs, q)
		rhogas = pt.rho_gas(cs, omega, sigma_pl, dust_to_gas[idx])
		e_eq_vals[idx] = pt.e_eq(omega, sigma_pl, mp, Rp, C_D, vk, rhogas, vgas)

	from matplotlib import cm
	fig, axes = plt.subplots(figsize=(8,8))
	cmap = mpl.cm.get_cmap('jet', 10)
	cax = axes.pcolormesh(t_orbit_d, e_i(e_h_vals), np.flipud(np.rot90(m_iso_vals)), \
	                      norm=mpl.colors.LogNorm(), cmap=cmap)
	for idx in range(len(dust_to_gas)):
		axes.plot(t_orbit_d, e_eq_vals[idx], color='white')
	cb = fig.colorbar(cax)
	cb.set_label('Isolation Mass [g]')
	axes.set_xlabel(r'Orbital Period [d]')
	axes.set_ylabel(r'$\left< e^{2} \right>^{1/2}$')
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlim(np.min(t_orbit_d), np.max(t_orbit_d))
	axes.set_ylim(np.min(e_i(e_h_vals)), np.max(e_i(e_h_vals)))
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_gasdrag():
	file_str = 'figures/gasdrag.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	t_stop_vals = np.empty((len(dust_to_gas), len(t_orbit)))
	t_drift_vals = np.empty((len(dust_to_gas), len(t_orbit)))

	for idx in range(len(dust_to_gas)):
	    cs = pt.soundspeed(T0, mu, q, a_vals_au)
	    vgas = pt.v_gas((m_central*u.g).to(u.M_sun).value, a_vals, cs, q)
	    rhogas = pt.rho_gas(cs, omega, sigma_pl, dust_to_gas[idx])
	    e_std = pt.e_eq(omega, sigma_pl, mp, Rp, C_D, vk, rhogas, vgas)
	    t_stop = pt.t_stop(mp, Rp, C_D, rhogas, vgas)
	    t_stop_vals[idx] = t_stop
	    vr = 2*vgas/((t_stop*omega)+(1/(t_stop*omega)))
	    t_drift_vals[idx] = a_vals/vr

	fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
	for idx in range(len(dust_to_gas)):
	    axes[0].plot(t_orbit_d, t_stop_vals[idx]/t_orbit, label='d/g = '+str(dust_to_gas[idx]))
	    axes[1].plot(t_orbit_d, t_drift_vals[idx]/t_orbit)
	axes[0].set_xscale('log')
	axes[0].set_yscale('log')
	axes[0].set_xlabel('Orbital Period [d]')
	axes[1].set_xlabel('Orbital Period [d]')
	axes[0].set_ylabel('Stopping Time [orbits]')
	axes[1].set_ylabel('Radial Drift Time [orbits]')
	axes[1].set_xscale('log')
	axes[1].set_yscale('log')
	axes[0].legend()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_plot():
	file_str = 'figures/test.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	xmodel = np.linspace(0, 10)
	ymodel = np.cos(xmodel)

	fig, axes = plt.subplots(figsize=(8,8))
	axes.plot(xmodel, ymodel)
	axes.set_xlabel('X')
	axes.set_ylabel('Y')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

#plot_trel_tcoll_eq_gf()
plot_m_iso()
#plot_gasdrag()