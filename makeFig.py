#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pynbody as pb
import natsort as ns
import glob as gl
import re
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
mCentralg = (mCentral*u.M_sun).to(u.g)

bins = np.linspace(0.01, 0.3, num_bins)

snap = pb.load('data/fullDiskVHi.ic')
plVHiIC = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic = pb.analysis.profile.Profile(plVHiIC.d, bins=bins)
snap = pb.load('data/fullDiskVHi1.348000')
plVHi = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi = pb.analysis.profile.Profile(plVHi.d, bins=bins)
perIC = 2*np.pi*np.sqrt((plVHiIC['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perIC = 2*np.pi*np.sqrt((p_vhi_ic['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
per = 2*np.pi*np.sqrt((plVHi['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

m_pl = np.min(plVHi['mass'])
# For some reason, the 'eps' field doesnt read correctly, so I can't
# pull the planeteismal radius from the snapshots
rho = 3 # Rocky bodies
r_pl = ((3*(m_pl*u.M_sun).to(u.g).value/(4*np.pi*rho))**(1/3)*u.cm).to(u.AU).value

snap = pb.load('data/fullDiskVHif4.ic')
plVHif4IC = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhif4_ic = pb.analysis.profile.Profile(plVHif4IC.d, bins=bins)
snap = pb.load('data/fullDiskVHif41.152000')
plVHif4 = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhif4 = pb.analysis.profile.Profile(plVHif4.d, bins=bins)
perf4IC = 2*np.pi*np.sqrt((plVHif4IC['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perf4IC = 2*np.pi*np.sqrt((p_vhif4_ic['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perf4 = 2*np.pi*np.sqrt((plVHif4['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskVHiSteep.ic')
plVHiICSt = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic_st = pb.analysis.profile.Profile(plVHiICSt.d, bins=bins)
snap = pb.load('data/fullDiskVHiSteep1.348000')
plVHiSt = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_st = pb.analysis.profile.Profile(plVHiSt.d, bins=bins)
perICSt = 2*np.pi*np.sqrt((plVHiICSt['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perIC_st = 2*np.pi*np.sqrt((p_vhi_ic_st['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perSt = 2*np.pi*np.sqrt((plVHiSt['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskVHiShallow.ic')
plVHiICSh = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic_sh = pb.analysis.profile.Profile(plVHiICSh.d, bins=bins)
snap = pb.load('data/fullDiskVHiShallow1.348000')
plVHiSh = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_sh = pb.analysis.profile.Profile(plVHiSh.d, bins=bins)
perICSh = 2*np.pi*np.sqrt((plVHiICSh['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perIC_sh = 2*np.pi*np.sqrt((p_vhi_ic_sh['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perSh = 2*np.pi*np.sqrt((plVHiSh['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskLo.ic')
plLoIC = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_lo_ic = pb.analysis.profile.Profile(plLoIC.d, bins=bins)
snap = pb.load('data/fullDiskLo1.426000')
plLo = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_lo = pb.analysis.profile.Profile(plLo.d, bins=bins)
perICLo = 2*np.pi*np.sqrt((plLoIC['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perIC_lo = 2*np.pi*np.sqrt((p_lo_ic['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perLo = 2*np.pi*np.sqrt((plLo['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

p_min, p_max = -5, 100
e_min, e_max = 1e-3, 1

def prev_iorder(new_iord, num_original, del_iord):
    indices = np.zeros(num_original, dtype=bool)
    indices[del_iord] = True
    prev_iord = np.where(indices == False)[0][new_iord]
    return prev_iord

# Input in sim units, output in days
def p_orbit(sma):
	return (2*np.pi*np.sqrt(sma**3/mCentral)*simT).to(u.d).value

def plot_timescales():
	file_str = 'figures/timescales.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	# r in AU
	def surf_den(r):
		sigma0 = 10 # g/cm^2
		alpha = 3/2
		return (sigma0*(r)**(-alpha)).value*u.g/u.cm**2

	rhop = 3*u.g/u.cm**3
	rp = (100*u.km).to(u.cm) # Timescale ratio is independent of size at fixed density
	mp = 4/3*np.pi*rhop*rp**3

	mstar = (1*u.M_sun).to(u.g)
	p_bins = (np.logspace(-4, 0)*u.yr).to(u.s)
	r_bins = ((p_bins/(2*np.pi))**2*G.cgs*mstar)**(1/3)

	def timescale_ratio(eh):
		Sigma = surf_den(r_bins.to(u.AU))

		vk = np.sqrt(G.cgs*mstar/r_bins)
		lnLam = 5
		omega = 2*np.pi/p_bins
		vesc = np.sqrt(G.cgs*mp/rp)

		# Dispersion dominated boundary
		ecc = eh*(mp/(3*mstar))**(1/3)
		inc = ecc/2
		sigma = np.sqrt(5/8*ecc**2 + inc**2)*vk

		n = Sigma*omega/(2*mp*sigma)
		theta = vesc/sigma
		cross = np.pi*rp**2*(1 + theta**2)
		t_coll_vals = 1/(n*cross*sigma)
		t_relax_vals = sigma**3/(n*np.pi*G.cgs**2*mp**2*lnLam)

		return t_relax_vals/t_coll_vals

	fig, axes = plt.subplots(figsize=(8,8))
	ehvals = [16, 8, 4, 2, 1]
	colors = plt.cm.viridis(np.linspace(0, 1, len(ehvals)))
	for idx, eh in enumerate(ehvals):
		axes.plot(p_bins.to(u.d), timescale_ratio(eh), c=colors[idx], label=r'e$_{h}$ = '+str(eh))
	axes.axhline(1, ls='--')
	axes.legend()
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r't$_{relax}$/t$_{coll}$')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_beta():
	file_str = 'figures/alpha_beta.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	def subplot(ax, prefix, title):
		mcen = 1
		snap0 = pb.load('data/'+prefix+'.ic')
		pl0 = ko.orb_params(snap0, isHelio=True, mCentral=mcen)
		snap = pb.load('data/'+prefix+'.end')
		pl = ko.orb_params(snap, isHelio=True, mCentral=mcen)
		print()
		ax.scatter(pl0['a'], pl0['e'], s=pl0['mass']/np.max(pl['mass'])*100)
		ax.scatter(pl['a'], pl['e'], s=pl['mass']/np.max(pl['mass'])*100)
		ax.set_xlabel('Semimajor Axis [AU]')
		ax.set_ylabel('Eccentricity')
		ax.set_title(title)

	fig, ax = plt.subplots(figsize=(16,16), nrows=2, ncols=2, sharex=True, sharey=True)
	subplot(ax[0][0], 'ki_fluffy', r'Large $\alpha$, Small $\beta$')
	subplot(ax[0][1], 'ki_fluffy_hot', r'Large $\alpha$, Large $\beta$')
	subplot(ax[1][0], 'ki', r'Small $\alpha$, Small $\beta$')
	subplot(ax[1][1], 'ki_hot', r'Small $\alpha$, Large $\beta$')

	ax[1][1].set_yscale('log')
	ax[1][1].set_xlim(0.9, 1.1)
	ax[1][1].set_ylim(1e-5, 0.3)

	plt.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_beta_evo():
	file_str = 'figures/alpha_beta_evo.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,8))

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_fluffy.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Large $\alpha$, Small $\beta$')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_fluffy_hot.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Large $\alpha$, Large $\beta$')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Small $\alpha$, Small $\beta$')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_hot.txt')
	ind = np.arange(len(stepnumber))
	stepnumber[ind >= 51] += 500000
	axes.loglog(stepnumber, max_mass/mean_mass, label=r'Small $\alpha$, Large $\beta$')

	axes.set_xlabel('Time Steps')
	axes.set_ylabel(r'M / $\langle$ m $\rangle$')
	axes.legend()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_beta_mass():
	file_str = 'figures/alpha_beta_mass.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,8), nrows=4, sharex=True, sharey=True)

	snap = pb.load('data/ki_fluffy.end')
	pl = ko.orb_params(snap, isHelio=True, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.g).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[0].loglog(bins, hist, linestyle='steps-mid')
	axes[0].set_title(r'Large $\alpha$, Small $\beta$')

	snap = pb.load('data/ki_fluffy_hot.end')
	pl = ko.orb_params(snap, isHelio=True, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.g).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[1].loglog(bins, hist, linestyle='steps-mid')
	axes[1].set_title(r'Large $\alpha$, Large $\beta$')

	snap = pb.load('data/ki.end')
	pl = ko.orb_params(snap, isHelio=False, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.g).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[2].loglog(bins, hist, linestyle='steps-mid')
	axes[2].set_title(r'Small $\alpha$, Small $\beta$')

	snap = pb.load('data/ki_hot.end')
	pl = ko.orb_params(snap, isHelio=True, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.g).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[3].loglog(bins, hist, linestyle='steps-mid')
	axes[3].set_title(r'Small $\alpha$, Large $\beta$')

	axes[2].set_ylabel('dn/dm')
	axes[3].set_xlabel('Mass [g]')

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
	btilde = 5
	m_iso_vhi = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentralg))
	btilde = 10
	m_iso_vhi1 = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentralg))

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

	fig, axes = plt.subplots(figsize=(16,8))

	mask1 = plVHi['mass'] < 10*np.min(plVHi['mass'])
	mask2 = plVHi['mass'] > 0

	bins1 = np.linspace(0.01, 0.3, 15)

	p_vhi_m1 = pb.analysis.profile.Profile(plVHi[mask1], bins=bins1)
	p_vhi_m2 = pb.analysis.profile.Profile(plVHi[mask2], bins=bins1)

	prof_per1 = 2*np.pi*np.sqrt((p_vhi_m1['rbins']*u.AU).to(u.cm)**3/(G.cgs*(mCentral*u.M_sun).to(u.g))).to(u.d)

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
	ax.set_ylim(-0.1, 1)

	axes.axvline(60, ls='--')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_pl_frac_time():
	file_str = 'figures/pl_frac_time.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	files = gl.glob1('data/', 'fullDiskVHi1.[0-9]*[0-9]')
	files = ns.natsorted(files)

	pbins = (np.linspace(1, 100, 10)*u.d).value
	bins1 = ((((pbins*u.d).to(u.s)/(2*np.pi))**2*G.cgs*(mCentral*u.M_sun).to(u.g))**(1/3)).to(u.AU).value

	files_sub = files

	prof_arr = np.zeros((len(bins1)-1, len(files_sub)))
	time_arr = np.zeros(len(files_sub))

	for idx, f in enumerate(files_sub):
		snap = pb.load('data/'+f)
		plVHi = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
		mask1 = plVHi['mass'] < 10*np.min(plVHi['mass'])
		mask2 = plVHi['mass'] > 0
		p_vhi_m1 = pb.analysis.profile.Profile(plVHi[mask1], bins=bins1)
		p_vhi_m2 = pb.analysis.profile.Profile(plVHi[mask2], bins=bins1)

		time_arr[idx] = snap.properties['time'].in_units('yr')
		prof_arr[:,idx] = (p_vhi_m1['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)/(p_vhi_m2['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)

	fig, axes = plt.subplots(figsize=(16,8))

	for idx in range(len(bins1[:-1])):
		if pbins[idx] < 60:
			ls = '--'
		else:
			ls = '-'
		axes.plot(time_arr, prof_arr[idx], color='black', linestyle=ls)

	axes.set_ylabel(r'$\sigma$ / $\Sigma$')
	axes.set_xlabel('Time [yr]')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_surfden_profiles():
	file_str = 'figures/surfden_profiles.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(16,8))

	axes.scatter(per.value, plVHi['e'], s=0.005*plVHi['mass']/np.min(plVHi['mass']), label='fdHi')
	axes.scatter(perSt.value, plVHiSt['e'], s=0.005*plVHiSt['mass']/np.min(plVHiSt['mass']), label='fdHiSteep')
	axes.scatter(perSh.value, plVHiSh['e'], s=0.005*plVHiSh['mass']/np.min(plVHiSh['mass']), label='fdHiShallow')
	axes.scatter(perLo.value, plLo['e'], s=0.005*plLo['mass']/np.min(plLo['mass']), label='fdLo')

	axes.set_yscale('log')
	axes.set_ylim(1e-3, 0.5)
	axes.set_xlim(-5, 100)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel('Eccentricity')
	axes.legend(loc=3)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_surfden_iso():
	file_str = 'figures/surfden_iso.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(16,8))

	btilde = 5

	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHi['a'], p_vhi_ic['rbins'], surf_den)
	m_iso_vhi_at = np.sqrt((2*np.pi*(plVHi['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHi')

	surf_den = (p_vhi_ic_st['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHiSt['a'], p_vhi_ic_st['rbins'], surf_den)
	m_iso_vhist_at = np.sqrt((2*np.pi*(plVHiSt['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perSt.value, (plVHiSt['mass']*u.M_sun).to(u.g).value/m_iso_vhist_at, label='fdHiSteep')

	surf_den = (p_vhi_ic_sh['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHiSh['a'], p_vhi_ic_sh['rbins'], surf_den)
	m_iso_vhish_at = np.sqrt((2*np.pi*(plVHiSh['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perSh.value, (plVHiSh['mass']*u.M_sun).to(u.g).value/m_iso_vhish_at, label='fdHiShallow')

	surf_den = (p_lo_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plLo['a'], p_lo_ic['rbins'], surf_den)
	m_iso_lo_at = np.sqrt((2*np.pi*(plLo['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perLo.value, (plLo['mass']*u.M_sun).to(u.g).value/m_iso_lo_at, label='fdLo')

	axes.axhline(1, ls='--')
	axes.set_yscale('log')
	axes.set_ylim(1e-2, 20)
	axes.set_xlim(-5, 100)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'm / M$_{iso}$')
	axes.legend()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

# Extract information about the 'surviving' members from a collision table
def coll_info(df):
    mass_mask = df['m1'].values >= df['m2'].values
    
    del_x = df['x1x'].values
    del_x[mass_mask] = df['x2x'].values[mass_mask]
    
    del_y = df['x1y'].values
    del_y[mass_mask] = df['x2y'].values[mass_mask]
    
    del_z = df['x1z'].values
    del_z[mass_mask] = df['x2z'].values[mass_mask]
    
    del_vx = df['v1x'].values
    del_vx[mass_mask] = df['v2x'].values[mass_mask]
    
    del_vy = df['v1y'].values
    del_vy[mass_mask] = df['v2y'].values[mass_mask]
    
    del_vz = df['v1z'].values
    del_vz[mass_mask] = df['v2z'].values[mass_mask]
    
    del_a, del_e, del_inc, del_omega, del_Omega, del_M = ko.cart2kep(del_x, del_y, del_z, del_vx, del_vy, del_vz, 0.08, 1e-20)
    
    del_r = np.sqrt(del_x**2 + del_y**2, del_z**2)
    
    dvx = df['v1x'].values - df['v2x'].values
    dvy = df['v1y'].values - df['v2y'].values
    dvz = df['v1z'].values - df['v2z'].values
    speed = np.sqrt(dvx**2 + dvy**2 + dvz**2)
    vesc = np.sqrt((df['m1'].values+df['m2'].values)/(df['r1'].values + df['r2'].values))
    
    return {'a': del_a, 'e': del_e, 'inc': del_inc, 'omega': del_omega, 'Omega': del_Omega, 'M': del_M, \
            'r': del_r,}

def plot_smooth_acc():
	file_str = 'figures/minor_frac.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	# For each embryo, fraction of mass accreted in planetesimals
	num_oli = 100

	df = pd.read_csv('data/fullDiskVHi.coll')
	df1_s, df1_d = np.loadtxt('data/fullDiskVHi_delete1', dtype='int', unpack=True)
	massive_ind = np.argsort(np.array(plVHi['mass']))[::-1][:num_oli]
	a_iord_orig = prev_iorder(plVHi['iord'][massive_ind], len(plVHiIC), df1_d)
	mmin = np.min(df['m1'])

	info = coll_info(df)
	p_coll = p_orbit(info['a'])

	xvals = np.zeros(num_oli)
	yvals = np.zeros_like(xvals)
	porb = np.zeros_like(xvals)

	for oli_idx in range(int(num_oli)):
		oli_iord = a_iord_orig[oli_idx]
		porbit = p_orbit(plVHi['a'][massive_ind[oli_idx]])

		mask = np.logical_or(df['iorder1'] == oli_iord, df['iorder2'] == oli_iord)
		mask1 = np.logical_or(df['m1'][mask] <= mmin, df['m2'][mask] <= mmin)
		frac = len(df[mask][mask1])*mmin/plVHi['mass'][massive_ind[oli_idx]]

		xvals[oli_idx] = (plVHi['mass'][massive_ind[oli_idx]]*u.M_sun).to(u.M_earth).value
		yvals[oli_idx] = frac
		porb[oli_idx] = porbit

	fig, axes = plt.subplots(figsize=(8,8))
	mask = porb < 60
	axes.scatter(xvals[~mask], yvals[~mask], color='r', marker='x', label='P_orbit > 60d')
	axes.scatter(xvals[mask], yvals[mask], label='P_orbit < 60d')
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_ylim(1e-4, 1e-1)
	axes.set_xlim(1e-3, 2)
	axes.set_xlabel('Mass [M_earth]')
	axes.set_ylabel('Smooth Accretion Mass Fraction')
	axes.legend()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_acc_zones():
	file_str = 'figures/acc_zones.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8, 8))

	s = 1000
	amin, amax = 0.05, 0.2
	histbins = np.linspace(0, 100, 200)

	root_idx = np.loadtxt('data/roots_vhi.txt')
	df1_s, df1_d = np.loadtxt('data/fullDiskVHi_delete1', dtype='int', unpack=True)
	massive_ind = np.argsort(np.array(plVHi['mass']))[::-1]
	massive_iord_orig = prev_iorder(plVHi['iord'][massive_ind], len(plVHiIC), df1_d)

	def p_orbit(a):
		return ((np.sqrt(a**3/mCentral))*u.yr).to(u.d).value

	def oli_plot(i, idx):
		child_ind = np.argwhere(root_idx == massive_iord_orig[i])
		child_a = plVHiIC['a'][child_ind]
		child_p = p_orbit(child_a)
		hist, bins = np.histogram(child_p, bins=histbins, normed=True)
		bins = 0.5*(bins[1:] + bins[:-1])
		hist /= np.max(hist)/0.1
		axes.plot(bins, hist+0.2*idx, linestyle='steps-mid', color='gray', lw=0.5)
		axes.fill_between(bins, np.min(hist)+0.2*idx, hist+0.2*idx)
		pl_p = p_orbit(plVHi['a'][massive_ind[i]])
		axes.vlines(pl_p, np.min(hist)+0.2*idx, np.max(hist)+0.2*idx )
		axes.set_xlabel('Orbital Period [d]')
		axes.set_ylabel('Planetesimals Accreted')

	# Sort by semimajor axis beore plotting
	indices = np.arange(0, 20)
	sort_a_ind = np.argsort(plVHi['a'][massive_ind[indices]])

	for idx, i in enumerate(sort_a_ind):
		oli_plot(i, idx)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_f6f4():
	file_str = 'figures/f6f4.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,8), nrows=2)

	axes = ax[0]
	btilde = 5

	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHi['a'], p_vhi_ic['rbins'], surf_den)
	m_iso_vhi_at = np.sqrt((2*np.pi*(plVHi['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHi')

	surf_den = (p_vhif4_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHif4['a'], p_vhif4_ic['rbins'], surf_den)
	m_iso_vhi_at = np.sqrt((2*np.pi*(plVHif4['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perf4.value, (plVHif4['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHif4')

	axes.axhline(1, ls='--')
	axes.set_yscale('log')
	axes.set_ylim(1e-2, 20)
	axes.set_xlim(-5, 100)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'm / M$_{iso}$')
	axes.legend()

	axes = ax[1]
	f = 6
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha)

	f = 4
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha)

	f = 1
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha)

	axes.axhline(0.1, ls='--')
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'$\alpha$')

	fig.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_frag_ecc():
	file_str = 'figures/frag_ecc.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	snap0 = pb.load('data/innerDiskLo.ic')
	pl0 = ko.orb_params(snap0, isHelio=True, mCentral=mCentral)
	per0 = 2*np.pi*np.sqrt((pl0['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/innerDiskLoNoFrag.6911000')
	plNf = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perNf = 2*np.pi*np.sqrt((plNf['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/innerDiskLoFrag.5676000')
	plF = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perF = 2*np.pi*np.sqrt((plF['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

	fig, ax = plt.subplots(figsize=(8,8), nrows=2)
	axes = ax[0]
	axes.scatter(per0, pl0['e'], s=pl0['mass']/np.min(pl0['mass'])*0.1)
	axes.scatter(perNf, plNf['e'], s=plNf['mass']/np.min(pl0['mass'])*1)
	axes.scatter(perF, plF['e'], s=plF['mass']/np.min(pl0['mass'])*1)
	axes.set_yscale('log')
	axes.set_ylim(5e-5, 5e-2)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel('Eccentricity')

	axes = ax[1]
	q = (plNf['mass']*u.M_sun).to(u.g).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes.loglog(bins, hist, linestyle='steps-mid')

	q = (plF['mass']*u.M_sun).to(u.g).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes.semilogx(bins, hist, linestyle='steps-mid')

	axes.set_xlabel('Mass [g]')
	axes.set_ylabel('dn/dm')

	fig.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_frag_evo():
	file_str = 'figures/frag_evo.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,8))

	stepnumber, max_mass, mean_mass = np.loadtxt('data/innerDiskLoNoFrag.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label='Mergers Only')

	stepnumber, max_mass, mean_mass = np.loadtxt('data/innerDiskLoFrag.txt')
	axes.loglog(stepnumber, max_mass/mean_mass, label='Bounce + Mergers')

	axes.set_xlabel('Time Steps')
	axes.set_ylabel(r'M / $\langle$ m $\rangle$')
	axes.legend()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

#plot_timescales()
#plot_alpha_beta()
#plot_alpha_beta_evo()
#plot_alpha_beta_mass()
#plot_fulldisk_e_m()
#plot_alpha_pl_frac()
#plot_pl_frac_time()
#plot_surfden_profiles()
#plot_surfden_iso()
#plot_smooth_acc()
#plot_acc_zones()
#plot_f6f4()
#plot_frag_ecc()
plot_frag_evo()