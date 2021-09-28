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
import KeplerOrbit as ko

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

simT = u.year/(2*np.pi)
simV = u.AU/simT

clobber = True
fmt = 'png'
s = 0.005

num_bins = 50
mCentral = 0.08

bins = np.linspace(0.01, 0.3, num_bins)
snap = pb.load('data/fullDiskVHi.ic')
plVHiIC = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic = pb.analysis.profile.Profile(plVHiIC.d, bins=bins)

snap = pb.load('data/fullDiskVHi.258000')
plVHi = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi = pb.analysis.profile.Profile(plVHi.d, bins=bins)

perIC = 2*np.pi*np.sqrt((plVHiIC['a']*u.AU).to(u.cm)**3/(G.cgs*(mCentral*u.M_sun).to(u.g))).to(u.d)
prof_perIC = 2*np.pi*np.sqrt((p_vhi_ic['rbins']*u.AU).to(u.cm)**3/(G.cgs*(mCentral*u.M_sun).to(u.g))).to(u.d)
per = 2*np.pi*np.sqrt((plVHi['a']*u.AU).to(u.cm)**3/(G.cgs*(mCentral*u.M_sun).to(u.g))).to(u.d)

p_min, p_max = -5, 100
e_min, e_max = 1e-3, 1

# Input in sim units, output in days
def p_orbit(sma, m_central):
	return (2*np.pi*np.sqrt(sma**3/m_central)*simT).to(u.d).value

def plot_alpha_beta():
	file_str = 'figures/alpha_beta.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	def subplot(ax, prefix, title):
		mcen = 1
		ih = True
		if prefix == 'disk4000':
			ih = False
		snap0 = pb.load('data/'+prefix+'.ic')
		pl0 = ko.orb_params(snap0, isHelio=ih, mCentral=mcen)
		snap = pb.load('data/'+prefix+'.end')
		pl = ko.orb_params(snap, isHelio=ih, mCentral=mcen)
		ax.scatter(p_orbit(pl0['a'], mcen), pl0['e'], s=pl0['mass']/np.max(pl['mass'])*100)
		ax.scatter(p_orbit(pl['a'], mcen), pl['e'], s=pl['mass']/np.max(pl['mass'])*100)
		ax.set_xlabel('Orbital Period [d]')
		ax.set_ylabel('Eccentricity')
		ax.set_title(title)

	fig, ax = plt.subplots(figsize=(16,16), nrows=2, ncols=2, sharex=True, sharey=True)
	subplot(ax[0][0], 'ki_fluffy_cold', r'Large $\alpha$, Small $\beta$')
	subplot(ax[0][1], 'ki_fluffy_superhot', r'Large $\alpha$, Large $\beta$')
	subplot(ax[1][0], 'disk4000', r'Small $\alpha$, Small $\beta$')
	subplot(ax[1][1], 'ki_superhot', r'Small $\alpha$, Large $\beta$')

	ax[1][1].set_yscale('log')
	ax[1][1].set_xlim(330, 400)
	ax[1][1].set_ylim(1e-5, 0.3)

	plt.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_beta_evo():
	file_str = 'figures/alpha_beta_evo.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,8))

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_fluffy_cold.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Large $\alpha$, Small $\beta$')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_fluffy_superhot.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Large $\alpha$, Large $\beta$')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/disk4000.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Small $\alpha$, Small $\beta$')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_superhot.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Small $\alpha$, Large $\beta$')

	axes.set_xlabel('Time Steps')
	axes.set_ylabel(r'M / $\langle$ m $\rangle$')
	axes.legend()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_fulldisk_e_m():
	file_str = 'figures/fulldisk_e_m.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(16,16), nrows=2, sharex=True)

	axes = ax[0]
	xbins = np.linspace(p_min, p_max, num=num_bins)
	ybins = np.logspace(np.log10(e_min), np.log10(e_max), num=num_bins)
	H, xedges, yedges = np.histogram2d(perIC.value, plVHiIC['e'], bins=(xbins, ybins))
	binsX = 0.5*(xedges[1:] + xedges[:-1])
	binsY = 0.5*(yedges[1:] + yedges[:-1])
	l = np.logspace(np.log10(30), np.log10(np.max(H)), 10)
	c = axes.contour(binsX, binsY, np.flipud(np.rot90(H)), colors='black', linewidths=0.2, levels=l)

	axes.scatter(per.value, plVHi['e'], s=0.005*plVHi['mass']/np.min(plVHi['mass']))
	axes.set_ylabel('Eccentricity')
	axes.set_xlim(p_min, p_max)
	axes.set_ylim(e_min, e_max)
	axes.set_yscale('log')

	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	a_vals = (p_vhi_ic['rbins']*u.AU).to(u.cm).value
	btilde = 2*np.sqrt(3)
	mCentral_g = (mCentral*u.M_sun).to(u.g).value
	m_iso_vhi = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentral_g))
	btilde = 10
	m_iso_vhi1 = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentral_g))

	axes = ax[1]
	axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value)
	axes.set_yscale('log')
	axes.plot(prof_perIC, m_iso_vhi)
	axes.plot(prof_perIC, m_iso_vhi1)
	axes.set_xlim(p_min, p_max)
	axes.set_ylim(1e25, 1e28)
	axes.set_ylabel('Mass [g]')
	axes.set_xlabel('Orbital Period [d]')
	axes.axvline(60, ls='--')

	plt.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_pl_frac():
	file_str = 'figures/alpha_pl_frac.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,4))

	mask1 = plVHi['mass'] < 10*np.min(plVHi['mass'])
	mask2 = plVHi['mass'] > 0

	bins1 = np.linspace(0.01, 0.3, 15)

	p_vhi_m1 = pb.analysis.profile.Profile(plVHi[mask1], bins=bins1)
	p_vhi_m2 = pb.analysis.profile.Profile(plVHi[mask2], bins=bins1)

	prof_per1 = 2*np.pi*np.sqrt((p_vhi_m1['rbins']*u.AU).to(u.cm)**3/(G.cgs*(mCentral*u.M_sun).to(u.g))).to(u.d)

	fig, axes = plt.subplots(figsize=(16,8))
	m0 = (np.min(plVHiIC['mass'])*u.M_sun).to(u.g).value
	rho = 3
	f = 6
	r0 = (3*m0/(4*np.pi*rho))**(1/3)
	alpha = (r0*u.cm*f).to(u.AU).value/(p_vhi_ic['a']*((m0*u.g).to(u.M_sun).value/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha)
	axes.set_ylim(5e-2, 1)
	axes.set_xlim(p_min, p_max)
	axes.set_ylabel(r'$\alpha$')
	axes.set_xlabel('Orbital Period [d]')

	ax = axes.twinx()
	ax.plot(prof_per1, (p_vhi_m1['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)/(p_vhi_m2['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2))
	ax.set_ylabel(r'$\sigma$ / $\Sigma$')
	ax.set_ylim(0, 1)

	axes.axvline(60, ls='--')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

#plot_alpha_beta()
#plot_alpha_beta_evo()
plot_fulldisk_e_m()
plot_alpha_pl_frac()