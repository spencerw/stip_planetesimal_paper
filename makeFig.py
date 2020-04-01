#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pynbody as pb
import natsort as ns
import glob as gl
import os

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
z = 1
h = 1
alpha = 1

rhop = 2
m_central = (0.14*u.M_sun).to(u.g).value
Rp = (50*u.km).to(u.cm).value
mp = 4/3*np.pi*Rp**3*rhop
f_pl = [1., 3., 6.]

# R in AU, returns units of g cm^-2
def surf_den(r, sigma1, f_disk, z, alpha):
    return sigma1*f_disk*z*r**(-alpha)*(m_central*u.g).to(u.M_sun).value**h

def e_i(e_h):
    return e_h*(mp/(3*m_central))**(1/3)

def e_h(e):
    return e*(mp/(3*m_central))**(-1/3)

a_vals_au = np.logspace(-2, 0.8)
e_h_vals = np.logspace(0, 2)

t_relax_vals = np.empty((len(a_vals_au), len(e_h_vals)))
t_coll_vals = np.empty((len(a_vals_au), len(e_h_vals)))
t_eq_eh_vals = np.empty((len(f_pl), len(a_vals_au)))

a_vals = (a_vals_au*u.AU).to(u.cm).value
t_orbit = (2*np.pi*np.sqrt((a_vals)**3/(G.cgs.value*m_central))*u.s).to(u.d).value

def plot_trel_tcoll_eq_gf():
	file_str = 'figures/time_eq_gf.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	for f_pl_idx in range(len(f_pl)):
	    for idx in range(len(a_vals)):
	        for idx1 in range(len(e_h_vals)):
	            e_std = e_i(e_h_vals[idx1])
	            i_std = e_std/2
	            Sigma = surf_den(a_vals_au[idx], sigma1, f_disk, z, alpha)
	            omega = np.sqrt(G.cgs.value*m_central/a_vals[idx]**3)
	            vk = np.sqrt(G.cgs.value*m_central/a_vals[idx])
	            sigma = np.sqrt(e_std**2 + i_std**2)*vk
	            n = Sigma*omega/(2*mp*sigma)

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
	            t_eq_eh_vals[f_pl_idx][idx] = e_h_vals[first_ind][0]
	        else:
	            t_eq_eh_vals[f_pl_idx][idx] = np.inf

	eh_gf_vals = np.empty((len(f_pl), len(a_vals)))

	for f_pl_idx in range(len(f_pl)):
	    for idx in range(len(a_vals)):
	        e_gf_vals = np.sqrt(4*mp*a_vals[idx]/(3*m_central*(f_pl[f_pl_idx]*Rp)))
	        eh_gf_vals[f_pl_idx][idx] = e_h(e_gf_vals)

	fig, axes = plt.subplots(figsize=(8,8))
	for idx, val in enumerate(f_pl):
	    c = next(axes._get_lines.prop_cycler)['color']
	    axes.plot(t_orbit, t_eq_eh_vals[idx], label=r'f$_{pl}$ = ' + str(int(val)), linestyle='-', color=c)
	    axes.plot(t_orbit, eh_gf_vals[idx], linestyle='--', color=c)
	axes.legend()
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlabel(r'Orbital Period [d]')
	axes.set_ylabel(r'Equilibirum $\left< \tilde{e}^{2} \right>^{1/2}$')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_m_iso():
	file_str = 'figures/m_iso.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	m_iso_vals = np.empty((len(a_vals_au), len(e_h_vals)))
	for idx in range(len(a_vals)):
	    for idx1 in range(len(e_h_vals)):
	        def func(x):
	            Sigma = surf_den(a_vals_au[idx], sigma1, f_disk, z, alpha)
	            e_std = e_i(e_h_vals[idx1])
	            i_std = e_std/2
	            return np.log10(4*np.pi*a_vals[idx]**2*Sigma*2/np.sqrt(3)*(10**x/(3*m_central))**(1/3)* \
	                            np.sqrt(9 + (10**x/(3*m_central))**(-2/3)*(e_std**2 + i_std**2))) - x
	        m_iso_vals[idx][idx1] = 10**optimize.root(func, 25).x

	from matplotlib import cm
	fig, axes = plt.subplots(figsize=(8,8))
	cmap = mpl.cm.get_cmap('jet', 10)
	cax = axes.pcolormesh(t_orbit, e_h_vals, np.flipud(np.rot90(m_iso_vals)), \
	                      norm=mpl.colors.LogNorm(), cmap=cmap)
	cb = fig.colorbar(cax)
	cb.set_label('Isolation Mass [g]')
	axes.set_xlabel(r'Orbital Period [d]')
	axes.set_ylabel(r'$\left< \tilde{e}^{2} \right>^{1/2}$')
	axes.set_xscale('log')
	axes.set_yscale('log')
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

plot_trel_tcoll_eq_gf()
plot_m_iso()