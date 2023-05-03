#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import colors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pynbody as pb
import natsort as ns
import glob as gl
import re
import os
import profileTools as pt
import pickle

from astropy import units as u
from astropy.constants import G
simT = u.year/(2*np.pi)
simV = u.AU / simT

from scipy import stats, optimize
import KeplerOrbit as ko
from profileTools import e_eq, rho_gas, v_gas, soundspeed, sma

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})
mpl.rcParams['mathtext.default'] = 'regular'

simT = u.year/(2*np.pi)
simV = u.AU/simT

clobber = True
fmt = 'png'
s = 0.005
lw = 4
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
refmass = (1*u.M_earth).to(u.M_sun).value

num_bins = 50
dDelta = 0.008213552361396302 # 2 pi years
mCentral = 0.08
mCentralg = (mCentral*u.M_sun).to(u.g)

bins = np.linspace(0.01, 0.3, num_bins)
perbins = 2*np.pi*np.sqrt((bins*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskVHia.ic')
plVHiIC = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic = pb.analysis.profile.Profile(plVHiIC.d, bins=bins, calc_x = lambda x: x['a'])
snap = pb.load('data/fullDiskVHi1.250000')
plVHi = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi = pb.analysis.profile.Profile(plVHi.d, bins=bins, calc_x = lambda x: x['a'])
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
p_vhif4_ic = pb.analysis.profile.Profile(plVHif4IC.d, bins=bins, calc_x = lambda x: x['a'])
snap = pb.load('data/fullDiskVHif41.152000')
plVHif4 = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhif4 = pb.analysis.profile.Profile(plVHif4.d, bins=bins, calc_x = lambda x: x['a'])
perf4IC = 2*np.pi*np.sqrt((plVHif4IC['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perf4IC = 2*np.pi*np.sqrt((p_vhif4_ic['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perf4 = 2*np.pi*np.sqrt((plVHif4['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskVHiSteep.ic')
plVHiICSt = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic_st = pb.analysis.profile.Profile(plVHiICSt.d, bins=bins, calc_x = lambda x: x['a'])
snap = pb.load('data/fullDiskVHiSteep1.348000')
plVHiSt = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_st = pb.analysis.profile.Profile(plVHiSt.d, bins=bins, calc_x = lambda x: x['a'])
perICSt = 2*np.pi*np.sqrt((plVHiICSt['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perIC_st = 2*np.pi*np.sqrt((p_vhi_ic_st['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perSt = 2*np.pi*np.sqrt((plVHiSt['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskVHiShallow.ic')
plVHiICSh = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_ic_sh = pb.analysis.profile.Profile(plVHiICSh.d, bins=bins, calc_x = lambda x: x['a'])
snap = pb.load('data/fullDiskVHiShallow1.348000')
plVHiSh = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_vhi_sh = pb.analysis.profile.Profile(plVHiSh.d, bins=bins, calc_x = lambda x: x['a'])
perICSh = 2*np.pi*np.sqrt((plVHiICSh['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
prof_perIC_sh = 2*np.pi*np.sqrt((p_vhi_ic_sh['rbins']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
perSh = 2*np.pi*np.sqrt((plVHiSh['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

snap = pb.load('data/fullDiskLoLarge.ic')
plLoIC = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_lo_ic = pb.analysis.profile.Profile(plLoIC.d, bins=bins, calc_x = lambda x: x['a'])
snap = pb.load('data/fullDiskLoLarge.3000000')
plLo = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
p_lo = pb.analysis.profile.Profile(plLo.d, bins=bins, calc_x = lambda x: x['a'])
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

# Determine the feeding zone widths required for a set of particles
# to obtain their current mass, based on an initial surface density profile
def get_btilde(pl, prof0):
	surf_den = (prof0['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	mp = (pl['mass']*u.M_sun).to(u.g).value
	mCeng = (mCentral*u.M_sun).to(u.g).value
	aCm = (pl['a']*u.AU).to(u.cm).value
	surf_den_at = np.interp(pl['a'], prof0['rbins'], surf_den)
	return mp**(2./3.)*(3*mCeng)**(1./3.)/(2*np.pi*aCm**2*surf_den_at)

# For a given snapshot, find a given property of the 'roots' of the particles
def get_root_property(pl0, pl, root, time, prop):
	def find(node, iord):
		if node.iord == iord:
			return node
		elif node.children:
			for c in node.children:
				result = find(c, iord)
				if result:
					return result

	root_iord = np.ones(len(pl0))*-1
	def get_roots_at(node, time, parent_iord):
		root_iord[node.iord] = parent_iord
		for c in node.children:
			if c.parent_time < time:
				get_roots_at(c, time, parent_iord)

	for iord in pl['iord']:
		node = find(root, iord)
		get_roots_at(node, time, node.iord)

	root_prop = []
	for i in range(len(pl)):
		child_ind = np.argwhere(root_iord == pl['iord'][i])
		root_prop.append(pl0[prop][child_ind])

	return root_prop

def plot_timescales():
	file_str = 'figures/timescales.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	# Values for M stars from Backus + Quinn 2016
	q = 0.59
	T0 = 130
	mu = 2.34 # Hayashi 1981

	C_D = 1

	def surf_den(r):
		sigma0 = 10 # g/cm^2
		alpha = 3/2
		return (sigma0*(r)**(-alpha)).value*u.g/u.cm**2

	rhop = 3*u.g/u.cm**3
	rp = (100*u.km).to(u.cm) # Timescale ratio is independent of size at fixed density
	mp = 4/3*np.pi*rhop*rp**3
	mstar = (1*u.M_sun).to(u.g)

	pvals = np.logspace(0, 2)
	pvals_s = (pvals*u.d).to(u.s)
	avals = sma(pvals_s.value, mstar.value)
	avals_au = (avals*u.cm).to(u.AU).value

	omega = 2*np.pi/pvals_s
	Sigma = surf_den(avals_au*u.AU)

	def calc_eqh(d_g):
		cs = soundspeed(T0, mu, q, avals_au)
		vgas = v_gas((mstar).to(u.M_sun).value, avals, cs, q)
		rhogas = rho_gas(cs, omega, Sigma, d_g)
		e_eq_vals = e_eq(omega, Sigma, mp, rp, C_D, vk, rhogas, vgas)
		return e_eq_vals/(mp/(3*mstar))**(1/3)

	fig, axes = plt.subplots(figsize=(8,6))

	ehvals = np.logspace(0, 2)
	trel_div_tcoll = np.zeros((len(pvals), len(ehvals)))
	for idx in range(len(ehvals)):

		vk = np.sqrt(G.cgs*mstar/(avals*u.cm))
		lnLam = 5
		omega = 2*np.pi/pvals_s
		vesc = np.sqrt(G.cgs*mp/rp)

		# Dispersion dominated boundary
		ecc = ehvals[idx]*(mp/(3*mstar))**(1/3)
		inc = ecc/2
		sigma = np.sqrt(5/8*ecc**2 + inc**2)*vk

		n = Sigma*omega/(2*mp*sigma)
		theta = vesc/sigma
		cross = np.pi*rp**2*(1 + theta**2)
		t_coll_vals = 1/(n*cross*sigma)
		t_relax_vals = sigma**3/(n*np.pi*G.cgs**2*mp**2*lnLam)

		trel_div_tcoll[idx] = t_relax_vals/t_coll_vals

	cmap = mpl.cm.get_cmap('seismic', 100)
	class MidPointLogNorm(mpl.colors.LogNorm):
		def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
			mpl.colors.LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
			self.midpoint=midpoint
		def __call__(self, value, clip=None):
			x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
			return np.ma.masked_array(np.interp(np.log(value), x, y))

	# Matplotlib bug: the midpoint on the colorbar has to be exactly in the center
	# of min and max, otherwise the colorbar doesnt match the colormap
	norm = MidPointLogNorm(vmin=np.min(1e-3), midpoint=1.0, vmax=np.max(1e3))
	cax = axes.pcolormesh(pvals, ehvals, np.flipud(np.rot90(trel_div_tcoll)), \
							norm=norm, cmap=cmap)
	axes.set_xscale('log')
	axes.set_yscale('log')

	cb = plt.colorbar(cax, ax=axes)
	cb.set_label(r'$t_{relax}/t_{collision}$')
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'$e_{h}$')

	pl_sizes = [50, 100, 200, 400]
	pl_sizes_cm = (pl_sizes*u.km).to(u.cm).value
	pl_masses_g = 3*4/3*np.pi*pl_sizes_cm**3
	labels = ['10 km', '50 km', '100 km', '200 km']
	m_pl_g = (m_pl*u.M_sun).to(u.g).value
	r_pl_cm = (r_pl*u.AU).to(u.cm).value
	for idx, size in enumerate(pl_sizes):
		v_k = np.sqrt(G.cgs.value*mstar.value/avals)
		v_esc = np.sqrt(2*G.cgs.value*pl_masses_g[idx]/pl_sizes_cm[idx])
		e_esc = v_esc/v_k
		yval = e_esc*(m_pl_g/(3*mstar.value))**(-1/3)
		axes.plot(pvals, yval, color='white', lw=4, ls='--')
		axes.text(1.5, yval[0]*0.9, labels[idx], rotation=15, color='white', ha='center', va='center')

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
		ax.scatter(pl0['a'], pl0['e'], s=pl0['mass']/refmass*1e4)
		ax.scatter(pl['a'], pl['e'], s=pl['mass']/refmass*1e4, edgecolor='black')
		ax.text(0.7, 0.95, title, transform=ax.transAxes, ha='center', va='center')

	fig, ax = plt.subplots(figsize=(16,12), nrows=2, ncols=2, sharex=True, sharey=True)
	subplot(ax[0][0], 'ki_fluffy', r'Case a: Large $\alpha$, Small $\beta$')
	subplot(ax[0][1], 'ki_fluffy_hot', r'Case b: Large $\alpha$, Large $\beta$')
	subplot(ax[1][0], 'ki', r'Case c: Small $\alpha$, Small $\beta$')
	ax[1][0].get_xticklabels()[0].set_visible(False)
	ax[1][0].get_xticklabels()[-1].set_visible(False)
	subplot(ax[1][1], 'ki_hot', r'Case d: Small $\alpha$, Large $\beta$')
	ax[1][1].get_xticklabels()[0].set_visible(False)
	ax[1][1].get_xticklabels()[-1].set_visible(False)

	ax[1][1].set_yscale('log')
	ax[1][1].set_xlim(0.9, 1.1)
	ax[1][1].set_ylim(1e-5, 0.3)

	l1 = plt.scatter([],[], edgecolor='black', color=orange, s=(1e-2*u.M_earth).to(u.M_sun).value/refmass*1e3)
	l2 = plt.scatter([],[], edgecolor='black', color=orange, s=(1e-1*u.M_earth).to(u.M_sun).value/refmass*1e3)
	l3 = plt.scatter([],[], edgecolor='black', color=orange, s=(1*u.M_earth).to(u.M_sun).value/refmass*1e3)
	labels = [r'0.01 M$_{\oplus}$', r'0.1 M$_{\oplus}$', r'1 M$_{\oplus}$']
	ax[0][0].legend(handles=[l1, l2, l3], labels=labels, loc=2)

	ax[0][0].set_ylabel(r'Large $\alpha$')
	ax[1][0].set_ylabel(r'Small $\alpha$')
	ax[1][0].set_xlabel(r'Small $\beta$')
	ax[1][1].set_xlabel(r'Large $\beta$')

	fig.supxlabel('Semimajor Axis [AU]')
	fig.supylabel('Eccentricity')
	
	plt.subplots_adjust(wspace=0, hspace=0)

	snap0 = pb.load('data/ki.ic')
	mp = (np.min(snap0['mass'])*u.M_sun).to(u.g).value
	rhop = 3
	rp = (3*mp/(4*np.pi*rhop))**(1/3)
	mstar = (1*u.M_sun).to(u.g).value
	e1 = 1.0*(mp/(3*mstar))**(1/3)
	vesc = np.sqrt(G.cgs.value*mp/rp)
	vk = np.sqrt(G.cgs.value*mstar/(1*u.AU).to(u.cm).value)
	e_esc = vesc/vk
	print('e_h = 1 corresponds to e = ' + str(e1))
	print('v = v_esc corresponds to e = ' + str(e_esc))

	mp = (0.8*u.M_earth).to(u.g).value
	rhop = 3
	rp = (3*mp/(4*np.pi*rhop))**(1/3)
	vesc = np.sqrt(G.cgs.value*mp/rp)
	vk = np.sqrt(G.cgs.value*0.08*mstar/(0.01*u.AU).to(u.cm).value)
	e_esc_pp = vesc/vk
	print('v = v_esc for protoplanets corresponds to e = ' + str(e_esc_pp))

	rhop = 3.0/7100
	rp = (3*mp/(4*np.pi*rhop))**(1/3)
	vesc = np.sqrt(G.cgs.value*mp/rp)
	e_esc = vesc/vk
	print('v = v_esc at high alpha corresponds to e = ' + str(e_esc))

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_beta_evo():
	file_str = 'figures/alpha_beta_evo.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,12), nrows=2, sharex=True)

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_fluffy.txt')
	c=next(axes[0]._get_lines.prop_cycler)['color']
	axes[0].loglog(stepnumber, max_mass/mean_mass, label=r'Case a: Large $\alpha$, Small $\beta$', lw=lw, color=c)
	axes[1].loglog(stepnumber, max_mass, lw=lw, color=c)
	axes[1].loglog(stepnumber, mean_mass, lw=lw, ls='--', color=c)

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_fluffy_hot.txt')
	c=next(axes[0]._get_lines.prop_cycler)['color']
	axes[0].loglog(stepnumber, max_mass/mean_mass, label=r'Case b: Large $\alpha$, Large $\beta$', lw=lw, color=c)
	axes[1].loglog(stepnumber, max_mass, lw=lw, color=c)
	axes[1].loglog(stepnumber, mean_mass, lw=lw, ls='--', color=c)

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki.txt')
	c=next(axes[0]._get_lines.prop_cycler)['color']
	axes[0].loglog(stepnumber, max_mass/mean_mass, label=r'Case c: Small $\alpha$, Small $\beta$', lw=lw, color=c)
	axes[1].loglog(stepnumber, max_mass, lw=lw, color=c)
	axes[1].loglog(stepnumber, mean_mass, lw=lw, ls='--', color=c)

	stepnumber, max_mass, mean_mass = np.loadtxt('data/ki_hot.txt')
	ind = np.arange(len(stepnumber))
	stepnumber[ind >= 51] += 500000
	c=next(axes[0]._get_lines.prop_cycler)['color']
	axes[0].loglog(stepnumber, max_mass/mean_mass, label=r'Case d: Small $\alpha$, Large $\beta$', lw=lw, color=c)
	axes[1].loglog(stepnumber, max_mass, lw=lw, color=c)
	axes[1].loglog(stepnumber, mean_mass, lw=lw, ls='--', color=c)

	axes[1].set_xlabel('Time Steps')
	axes[0].set_ylabel(r'M / $\langle$ m $\rangle$')
	axes[1].set_ylabel(r'M, $\langle$ m $\rangle$ [M$_{\oplus}$]')
	axes[0].set_xlim(5e3,8e7)
	axes[0].set_ylim(0.5, 300)
	axes[0].legend()

	plt.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_alpha_beta_mass():
	file_str = 'figures/alpha_beta_mass.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,6), nrows=4, sharex=True, sharey=True)

	snap = pb.load('data/ki_fluffy.end')
	pl = ko.orb_params(snap, isHelio=True, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[0].loglog(bins, hist, drawstyle='steps-mid', color=orange)
	axes[0].fill_between(bins, 0, hist, step='mid', alpha=0.5, color=orange)
	axes[0].set_yticks([])
	axes[0].text(0.7, 0.85, r'Case a: Large $\alpha$, Small $\beta$', transform=axes[0].transAxes, ha='center', va='center')

	snap = pb.load('data/ki_fluffy_hot.end')
	pl = ko.orb_params(snap, isHelio=True, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[1].loglog(bins, hist, drawstyle='steps-mid', color=orange)
	axes[1].fill_between(bins, 0, hist, step='mid', alpha=0.5, color=orange)
	axes[1].set_yticks([])
	axes[1].text(0.7, 0.85, r'Case b: Large $\alpha$, Large $\beta$', transform=axes[1].transAxes, ha='center', va='center')

	snap = pb.load('data/ki.end')
	pl = ko.orb_params(snap, isHelio=False, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[2].loglog(bins, hist, drawstyle='steps-mid', color=orange)
	axes[2].fill_between(bins, 0, hist, step='mid', alpha=0.5, color=orange)
	axes[2].set_yticks([])
	axes[2].text(0.7, 0.85, r'Case c: small $\alpha$, Small $\beta$', transform=axes[2].transAxes, ha='center', va='center')

	snap = pb.load('data/ki_hot.end')
	pl = ko.orb_params(snap, isHelio=True, mCentral=1)
	q = (pl['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q, bins=np.logspace(np.min(np.log10(q)), np.max(np.log10(q))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes[3].loglog(bins, hist, drawstyle='steps-mid', color=orange)
	axes[3].fill_between(bins, 0, hist, step='mid', alpha=0.5, color=orange)
	axes[3].set_yticks([])
	axes[3].text(0.7, 0.85, r'Case d: Small $\alpha$, Large $\beta$', transform=axes[3].transAxes, ha='center', va='center')

	fig.supylabel('log dn/dm')
	axes[3].set_xlabel(r'Mass [$M_{Earth}$]')

	plt.subplots_adjust(wspace=0, hspace=0)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_fulldisk_e_m():
	file_str = 'figures/fulldisk_e_m.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)

	axes = ax[0]
	# Not sure why I have to set the bin minimum this way, but it prevents a weird
	# artifact from appearing on the left side of the contours
	xbins = np.linspace(-1, p_max, num=num_bins)
	ybins = np.logspace(np.log10(e_min), np.log10(e_max), num=num_bins)
	H, xedges, yedges = np.histogram2d(perIC.value, plVHiIC['e'], bins=(xbins, ybins))
	binsX = 0.5*(xedges[1:] + xedges[:-1])
	binsY = 0.5*(yedges[1:] + yedges[:-1])
	l = np.logspace(np.log10(30), np.log10(np.max(H)), 10)
	c = axes.contour(binsX, binsY, np.flipud(np.rot90(H)), colors='black', linewidths=0.2, levels=l)

	axes.scatter(per.value, plVHi['e'], s=plVHi['mass']/refmass*500)
	axes.set_ylabel('Eccentricity')
	axes.set_xlim(p_min, p_max)
	axes.set_ylim(e_min, e_max)
	axes.set_yscale('log')

	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	a_vals = (p_vhi_ic['rbins']*u.AU).to(u.cm).value
	btilde = 2*np.sqrt(3)
	m_iso_vhi = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentralg))
	btilde = 10
	m_iso_vhi1 = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentralg))

	axes = ax[1]
	axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value, s=plVHi['mass']/refmass*500)
	axes.set_yscale('log')
	axes.plot(prof_perIC, m_iso_vhi, lw=lw)
	axes.plot(prof_perIC, m_iso_vhi1, lw=lw)
	axes.set_xlim(p_min, p_max)
	axes.set_ylim(1e25, 1e28)
	axes.set_ylabel('Mass [g]')
	axes.set_xlabel('Orbital Period [d]')
	axes.axvline(60, ls='--')

	plt.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_fulldisk_e_m_b():
	file_str = 'figures/fulldisk_e_m_b.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)

	axes = ax[0]
	# Not sure why I have to set the bin minimum this way, but it prevents a weird
	# artifact from appearing on the left side of the contours
	xbins = np.linspace(-1, p_max, num=num_bins)
	ybins = np.logspace(np.log10(e_min), np.log10(e_max), num=num_bins)
	H, xedges, yedges = np.histogram2d(perIC.value, plVHiIC['e'], bins=(xbins, ybins))
	binsX = 0.5*(xedges[1:] + xedges[:-1])
	binsY = 0.5*(yedges[1:] + yedges[:-1])
	l = np.logspace(np.log10(30), np.log10(np.max(H)), 10)
	c = axes.contour(binsX, binsY, np.flipud(np.rot90(H)), colors='black', linewidths=0.2, levels=l)

	axes.scatter(per.value, plVHi['e'], s=plVHi['mass']/refmass*1000, edgecolor='black', linewidth=0.4, color='#ff7f0e')
	axes.set_ylabel('Eccentricity')
	axes.set_xlim(p_min, p_max)
	axes.set_ylim(e_min, e_max)
	axes.set_yscale('log')

	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	a_vals = (p_vhi_ic['rbins']*u.AU).to(u.cm).value
	btilde = 2*np.sqrt(3)
	m_iso_vhi = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentralg))
	#btilde = 10
	#m_iso_vhi1 = np.sqrt((2*np.pi*a_vals**2*btilde*surf_den)**3/(3*mCentralg))

	axes = ax[1]
	axes.scatter(per.value, get_btilde(plVHi, p_vhi_ic), edgecolor='black', linewidth=0.4, color='#ff7f0e')
	axes.axhline(2.*np.sqrt(3.), ls='--', color='gray')
	#axes.axhline(10., ls='--', color='gray')
	axes.set_ylim(0, 20)
	axes.text(15, 2*np.sqrt(3)*1.1, r'$\tilde{b} = 2 \sqrt{3}$ (Circular Orbit)')
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'Required $\tilde{b}$')

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
	ax.plot(prof_per1, \
		   (p_vhi_m1['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)/(p_vhi_m2['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2),\
		   lw=lw)
	ax.set_ylabel(r'$\sigma$ / $\Sigma$')
	ax.set_ylim(-0.1, 1)

	axes.axvline(60, ls='--')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_pl_frac_time():
	file_str = 'figures/pl_frac_time.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	files = []#ns.natsorted(gl.glob1('data/','fullDiskVHi.[0-9]*[0-9]'))
	files1 = ns.natsorted(gl.glob1('data/', 'fullDiskVHi1.[0-9]*[0-9]'))
	files = np.concatenate([files, files1])

	pbins = (np.linspace(1, 100, 20)*u.d).value
	bins1 = ((((pbins*u.d).to(u.s)/(2*np.pi))**2*G.cgs*(mCentral*u.M_sun).to(u.g))**(1/3)).to(u.AU).value
	prof_ic = pb.analysis.profile.Profile(plVHiIC.d, bins=bins1, calc_x = lambda x: x['a'])

	log_ind = np.unique(np.logspace(0, np.log10(len(files)), 40).astype(np.int32))
	log_ind -= 1
	files_sub = files[log_ind]

	prof_arr = np.zeros((len(bins1)-1, len(files_sub)))
	time_arr = np.zeros(len(files_sub))

	for idx, f in enumerate(files_sub):
		snap = pb.load('data/'+f)
		plVHi = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
		mask = plVHi['mass'] <= 100*np.min(plVHi['mass'])
		p_vhi_m1 = pb.analysis.profile.Profile(plVHi[mask], bins=bins1)
		p_vhi_m2 = pb.analysis.profile.Profile(plVHiIC, bins=bins1)

		time = int(f.split('.')[-1])*dDelta/(2*np.pi)*365.25/100 # Outer disk rotation periods
		time_arr[idx] = time
		prof_arr[:,idx] = (p_vhi_m1['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)/(p_vhi_m2['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)

	fig, axes = plt.subplots(figsize=(8,4))
	col = cm.jet(np.linspace(0, 1, len(pbins)))
	for idx in range(len(bins1[:-1])):
		tdyn_loc = pbins[idx]/(2*np.pi)

		e_disp = prof_ic['e_disp'][idx]
		vk = np.sqrt(mCentral/prof_ic['rbins'][idx])
		v = np.sqrt(e_disp)*vk

		sigma_geo = np.pi*(6*r_pl)**2
		v_esc = np.sqrt(2*m_pl/(6*r_pl))
		sigma = sigma_geo*(1 + (v_esc**2/v**2))

		omega = np.sqrt(mCentral/prof_ic['rbins'][idx]**3)
		n = prof_ic['density'][idx]*omega/(2*m_pl*v)

		tacc_loc = 1/(n*sigma*v)

		mask = prof_arr[idx] > 0
		cbs = axes.plot(time_arr[mask][1:]/tacc_loc, prof_arr[idx][mask][1:], \
			lw=lw, color=col[idx])

	axes.set_ylabel(r'$\sigma$ / $\Sigma$')
	axes.set_xlabel('Time [Local Accretion Timescale]')
	axes.set_xscale('log')
	axes.set_xlim(4e-2, 40)

	norm = colors.Normalize(vmin=0, vmax=100)
	sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
	cb = plt.colorbar(sm)
	cb.set_label('Orbital Period [d]')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_surfden_profiles():
	file_str = 'figures/surfden_profiles.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,12), nrows=4, sharex=True, sharey=True)

	size = 500
	axes = ax[0]
	axes.set_title('fdHi')
	axes.scatter(per.value, plVHi['e'], s=size*plVHi['mass']/np.max(plVHi['mass']), label='fdHi', edgecolor='black', linewidth=0.4, color=orange)
	axes = ax[1]
	axes.set_title('fdSteep')
	axes.scatter(perSt.value, plVHiSt['e'], s=size*plVHiSt['mass']/np.max(plVHiSt['mass']), label='fdHiSteep', edgecolor='black', linewidth=0.4, color=orange)
	axes = ax[2]
	axes.set_title('fdShallow')
	axes.scatter(perSh.value, plVHiSh['e'], s=size*plVHiSh['mass']/np.max(plVHiSh['mass']), label='fdHiShallow', edgecolor='black', linewidth=0.4, color=orange)
	axes = ax[3]
	axes.set_title('fdLo')
	axes.scatter(perLo.value, plLo['e'], s=size*plLo['mass']/np.max(plLo['mass']), label='fdLo', edgecolor='black', linewidth=0.4, color=orange)

	axes.set_yscale('log')
	axes.set_ylim(1e-4, 0.5)
	axes.set_xlim(-5, 100)
	axes.set_xlabel('Orbital Period [d]')
	fig.supylabel('Eccentricity')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_surfden_iso():
	file_str = 'figures/surfden_iso.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,12), nrows=4, sharex=True, sharey=True)

	btilde = 2*np.sqrt(3)#5
	s = 50

	axes = ax[0]
	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHi['a'], p_vhi_ic['rbins'], surf_den)
	m_iso_vhi_at = np.sqrt((2*np.pi*(plVHi['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHi', s=s, edgecolor='black', linewidth=0.4, color=orange)
	axes.axhline(1, ls='--', color='gray')
	axes = ax[1]
	surf_den = (p_vhi_ic_st['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHiSt['a'], p_vhi_ic_st['rbins'], surf_den)
	m_iso_vhist_at = np.sqrt((2*np.pi*(plVHiSt['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perSt.value, (plVHiSt['mass']*u.M_sun).to(u.g).value/m_iso_vhist_at, label='fdHiSteep', s=s, edgecolor='black', linewidth=0.4, color=orange)
	axes.axhline(1, ls='--', color='gray')
	axes = ax[2]
	surf_den = (p_vhi_ic_sh['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHiSh['a'], p_vhi_ic_sh['rbins'], surf_den)
	m_iso_vhish_at = np.sqrt((2*np.pi*(plVHiSh['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perSh.value, (plVHiSh['mass']*u.M_sun).to(u.g).value/m_iso_vhish_at, label='fdHiShallow', s=s, edgecolor='black', linewidth=0.4, color=orange)
	axes.axhline(1, ls='--', color='gray')
	axes = ax[3]
	surf_den = (p_lo_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plLo['a'], p_lo_ic['rbins'], surf_den)
	m_iso_lo_at = np.sqrt((2*np.pi*(plLo['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perLo.value, (plLo['mass']*u.M_sun).to(u.g).value/m_iso_lo_at, label='fdLo', s=s, edgecolor='black', linewidth=0.4, color=orange)
	axes.axhline(1, ls='--', color='gray')
	axes.set_yscale('log')
	axes.set_ylim(1e-2, 20)
	axes.set_xlim(-5, 100)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'm / M$_{iso}$')
	#axes.legend(loc=3)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_surfden_b():
	file_str = 'figures/surfden_b.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,12), nrows=4, sharex=True, sharey=True)

	snap = pb.load('data/fullDiskVHi1b.248000')
	plVHib = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perb = 2*np.pi*np.sqrt((plVHib['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHi1c.248000')
	plVHic = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perc = 2*np.pi*np.sqrt((plVHic['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHi1d.248000')
	plVHid = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perd = 2*np.pi*np.sqrt((plVHid['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHi1e.248000')
	plVHie = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	pere = 2*np.pi*np.sqrt((plVHie['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

	snap = pb.load('data/fullDiskVHiSteep1b.324000')
	plVHiStb = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perStb = 2*np.pi*np.sqrt((plVHiStb['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHiSteep1c.324000')
	plVHiStc = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perStc = 2*np.pi*np.sqrt((plVHiStc['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHiSteep1d.324000')
	plVHiStd = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perStd = 2*np.pi*np.sqrt((plVHiStd['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHiSteep1e.324000')
	plVHiSte = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perSte = 2*np.pi*np.sqrt((plVHiSte['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

	snap = pb.load('data/fullDiskVHiShallow1b.313000')
	plVHiShb = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perShb = 2*np.pi*np.sqrt((plVHiShb['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHiShallow1c.313000')
	plVHiShc = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perShc = 2*np.pi*np.sqrt((plVHiShc['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHiShallow1d.313000')
	plVHiShd = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perShd = 2*np.pi*np.sqrt((plVHiShd['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/fullDiskVHiShallow1e.313000')
	plVHiShe = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perShe = 2*np.pi*np.sqrt((plVHiShe['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

	perbins1 = np.linspace(1, 100, 6)
	perbincents = 0.5*(perbins1[1:] + perbins1[:-1])


	periods = np.concatenate([per.value, perb.value, perc.value, perd.value, pere.value])
	btildes = np.concatenate([get_btilde(plVHi, p_vhi_ic), get_btilde(plVHib, p_vhi_ic),
		get_btilde(plVHic, p_vhi_ic), get_btilde(plVHid, p_vhi_ic), get_btilde(plVHie, p_vhi_ic)])
	periodsSt = np.concatenate([perSt.value, perStb.value, perStc.value, perStd.value, perSte.value])
	btildesSt = np.concatenate([get_btilde(plVHiSt, p_vhi_ic_st), get_btilde(plVHiStb, p_vhi_ic_st),
		get_btilde(plVHiStc, p_vhi_ic_st), get_btilde(plVHiStd, p_vhi_ic_st), get_btilde(plVHiSte, p_vhi_ic_st)])
	periodsSh = np.concatenate([perSh.value, perShb.value, perShc.value, perShd.value, perShe.value])
	btildesSh = np.concatenate([get_btilde(plVHiSh, p_vhi_ic_sh), get_btilde(plVHiShb, p_vhi_ic_sh),
		get_btilde(plVHiShc, p_vhi_ic_sh), get_btilde(plVHiShd, p_vhi_ic_sh), get_btilde(plVHiShe, p_vhi_ic_sh)])
	periodsLo = np.concatenate([perLo.value])
	btildesLo = np.concatenate([get_btilde(plLo, p_lo_ic)])

	axes = ax[0]
	axes.scatter(per.value, get_btilde(plVHi, p_vhi_ic), label='fdHi', edgecolor='black', linewidth=0.4)
	#axes.plot(perbincents, fdhi_means)
	#axes.fill_between(perbincents, fdhi_means - fdhi_std, fdhi_means + fdhi_std, color='blue', alpha=0.3)
	axes.scatter(periods, btildes, label='fdHi', edgecolor='black', linewidth=0.4, color=orange)
	#axes.scatter(perb.value, get_btilde(plVHib, p_vhi_ic), label='fdHi', edgecolor='black', linewidth=0.4)
	#axes.scatter(perc.value, get_btilde(plVHic, p_vhi_ic), label='fdHi', edgecolor='black', linewidth=0.4)
	#axes.scatter(perd.value, get_btilde(plVHid, p_vhi_ic), label='fdHi', edgecolor='black', linewidth=0.4)
	#axes.scatter(pere.value, get_btilde(plVHie, p_vhi_ic), label='fdHi', edgecolor='black', linewidth=0.4)
	axes.axhline(2*np.sqrt(3), ls='--', color='gray')
	#axes.axhline(10, ls='--', color='gray')
	axes.set_title('fdHi')
	axes = ax[1]
	#axes.scatter(perSt.value, get_btilde(plVHiSt, p_vhi_ic_st), label='fdSteep', edgecolor='black', linewidth=0.4)
	#axes.plot(perbincents, fdhiSt_means)
	#axes.fill_between(perbincents, fdhiSt_means - fdhiSt_std, fdhiSt_means + fdhiSt_std, color='blue', alpha=0.3)
	axes.scatter(periodsSt, btildesSt, label='fdSteep', edgecolor='black', linewidth=0.4, color=orange)
	#axes.scatter(perStb.value, get_btilde(plVHiStb, p_vhi_ic_st), label='fdSteep', edgecolor='black', linewidth=0.4)
	#axes.scatter(perStc.value, get_btilde(plVHiStc, p_vhi_ic_st), label='fdSteep', edgecolor='black', linewidth=0.4)
	#axes.scatter(perStd.value, get_btilde(plVHiStd, p_vhi_ic_st), label='fdSteep', edgecolor='black', linewidth=0.4)
	#axes.scatter(perSte.value, get_btilde(plVHiSte, p_vhi_ic_st), label='fdSteep', edgecolor='black', linewidth=0.4)
	axes.axhline(2*np.sqrt(3), ls='--', color='gray')
	#axes.axhline(10, ls='--', color='gray')
	axes.set_title('fdHiSteep')
	axes = ax[2]
	#axes.plot(perbincents, fdhiSh_means)
	#axes.fill_between(perbincents, fdhiSh_means - fdhiSh_std, fdhiSh_means + fdhiSh_std, color='blue', alpha=0.3)
	axes.scatter(periodsSh, btildesSh, label='fdShallow', edgecolor='black', linewidth=0.4, color=orange)
	#axes.scatter(perSh.value, get_btilde(plVHiSh, p_vhi_sh), label='fdShallow', edgecolor='black', linewidth=0.4)
	#axes.scatter(perShb.value, get_btilde(plVHiShb, p_vhi_sh), label='fdShallow', edgecolor='black', linewidth=0.4)
	#axes.scatter(perShc.value, get_btilde(plVHiShc, p_vhi_sh), label='fdShallow', edgecolor='black', linewidth=0.4)
	#axes.scatter(perShd.value, get_btilde(plVHiShd, p_vhi_sh), label='fdShallow', edgecolor='black', linewidth=0.4)
	#axes.scatter(perShe.value, get_btilde(plVHiShe, p_vhi_sh), label='fdShallow', edgecolor='black', linewidth=0.4)
	axes.axhline(2*np.sqrt(3), ls='--', color='gray')
	#axes.axhline(10, ls='--', color='gray')
	#rungs = np.arange(0, 7)
	#for rung in rungs:
		#axes.axvline(2**rung)
	axes.set_title('fdHiShallow')
	axes = ax[3]
	#axes.plot(perbincents, fdLo_means)
	#axes.fill_between(perbincents, fdLo_means - fdLo_std, fdLo_means + fdLo_std, color='blue', alpha=0.3)
	axes.scatter(perLo.value, get_btilde(plLo, p_lo_ic), label='fdLo', edgecolor='black', color=orange, linewidth=0.4)
	axes.axhline(2*np.sqrt(3), ls='--', color='gray')
	#axes.axhline(10, ls='--', color='gray')
	axes.set_title('fdLo')

	axes.set_xlim(-5, 100)
	#axes.set_yscale('log')
	axes.set_ylim(0, 20)
	axes.set_xlabel('Orbital Period [d]')
	#axes.legend()

	fig.supylabel(r'Required $\tilde{b}$')

	plt.tight_layout()
	#plt.subplots_adjust(wspace=0, hspace=0)
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

	del_mass = df['m1'].values
	del_mass[mass_mask] = df['m2'].values[mass_mask]
    
	dvx = df['v1x'].values - df['v2x'].values
	dvy = df['v1y'].values - df['v2y'].values
	dvz = df['v1z'].values - df['v2z'].values
	speed = np.sqrt(dvx**2 + dvy**2 + dvz**2)
	vesc = np.sqrt((df['m1'].values+df['m2'].values)/(df['r1'].values + df['r2'].values))

	return {'a': del_a, 'e': del_e, 'inc': del_inc, 'omega': del_omega, 'Omega': del_Omega, 'M': del_M, \
			'r': del_r, 'mass': del_mass}

def plot_smooth_acc():
	file_str = 'figures/minor_frac.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	nam = ['time', 'collType', 'iorder1', 'iorder2', 'm1', 'm2', 'r1', 'r2', 'x1x', 'x1y', 'x1z', 'x2x', 'x2y', 'x2z',  'v1x', 'v1y', 'v1z', \
	'v2x', 'v2y', 'v2z', 'w1x', 'w1y', 'w1z', 'w2x', 'w2y', 'w2z']

	fig, ax = plt.subplots(figsize=(8,18), nrows=3, sharex=True, sharey=True)
	ax[0].set_xscale('log')
	ax[0].set_yscale('log')
	ax[0].set_ylim(1e-2, 1)
	ax[0].set_xlim(1e-3, 2)

	def plot_smooth_sim(ax, simname, snapnum):
		print(simname)
		time = 0.05*snapnum/(2*np.pi)
		df1 = pd.read_csv('data/'+simname+'/fullDiskVHi.coll', names=nam, sep=' ', index_col=False)
		df2 = pd.read_csv('data/'+simname+'/fullDiskVHi1.coll', names=nam, sep=' ', index_col=False)
		df = pd.concat([df1, df2])
		df = df[df['time'] <= time]

		snap = pb.load('data/'+simname+'/fullDiskVHi1.'+str(snapnum))
		pl = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
		df1_d, df1_s = np.loadtxt('data/'+simname+'/delete1', dtype='int', unpack=True)
		massive_ind = np.argsort(np.array(pl['mass']))[::-1]
		massive_iord_orig = prev_iorder(pl['iord'][massive_ind], len(plVHiIC), df1_d)
		mmin = 100*np.min(df['m1'])

		info = coll_info(df)
		p_coll = p_orbit(info['a'])
		num_oli = len(pl)

		xvals = np.zeros(num_oli)
		yvals = np.zeros_like(xvals)
		porb = np.zeros_like(xvals)

		for oli_idx in range(int(num_oli)):
			oli_iord = massive_iord_orig[oli_idx]
			porbit = p_orbit(pl['a'][massive_ind[oli_idx]])

			mask = np.logical_or(df['iorder1'] == oli_iord, df['iorder2'] == oli_iord)
			mask1 = np.logical_or(df['m1'][mask] <= mmin, df['m2'][mask] <= mmin)
			frac = np.sum(info['mass'][mask][mask1])/pl['mass'][massive_ind[oli_idx]]

			xvals[oli_idx] = (pl['mass'][massive_ind[oli_idx]]*u.M_sun).to(u.M_earth).value
			yvals[oli_idx] = frac
			porb[oli_idx] = porbit

		import matplotlib.cm as cm
		cbs = ax.scatter(xvals, yvals, edgecolor='black', \
			linewidth=0.4, c=porb, cmap=cm.jet, vmin=0.0, vmax=100.0, marker='o')

		axes.set_ylabel('Smooth Accretion Mass Fraction')

		return cbs

	axes = ax[0]
	axes.set_title('fdHi')
	cbs = plot_smooth_sim(axes, 'fdvhia', 248000)
	cbs = plot_smooth_sim(axes, 'fdvhib', 248000)
	cbs = plot_smooth_sim(axes, 'fdvhic', 248000)
	cbs = plot_smooth_sim(axes, 'fdvhid', 248000)
	cbs = plot_smooth_sim(axes, 'fdvhie', 248000)
	cb = fig.colorbar(cbs, ax=axes)
	cb.set_label('Orbital Period [d]')

	axes = ax[1]
	axes.set_title('fdHiSteep')
	cbs = plot_smooth_sim(axes, 'fdvhisteepa', 345000)
	cbs = plot_smooth_sim(axes, 'fdvhisteepb', 324000)
	cbs = plot_smooth_sim(axes, 'fdvhisteepc', 329000)
	cbs = plot_smooth_sim(axes, 'fdvhisteepd', 327000)
	cbs = plot_smooth_sim(axes, 'fdvhisteepe', 348000)
	cb = fig.colorbar(cbs, ax=axes)
	cb.set_label('Orbital Period [d]')

	axes = ax[2]
	axes.set_title('fdHiShallow')
	cbs = plot_smooth_sim(axes, 'fdvhishallowa', 348000)
	cbs = plot_smooth_sim(axes, 'fdvhishallowb', 319000)
	cbs = plot_smooth_sim(axes, 'fdvhishallowc', 318000)
	cbs = plot_smooth_sim(axes, 'fdvhishallowd', 313000)
	cbs = plot_smooth_sim(axes, 'fdvhishallowe', 313000)
	cb = fig.colorbar(cbs, ax=axes)
	cb.set_label('Orbital Period [d]')

	axes.set_xlabel(r'Mass [M$_{\oplus}$]')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_acc_zones():
	file_str = 'figures/acc_zones.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8, 6))
	histbins = np.linspace(0.5, 1.5, 200)

	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'$\left( P - \left< P_{acc} \right> \right) / P_{acc}$')
	axes.axhline(0, ls='--', lw=lw, color='gray')
	axes.set_xlim(1, 100)
	axes.set_xscale('log')
	axes.set_ylim(-0.3, 0.3)

	def get_data(sim, snapnum, icfile):
		with open('data/'+sim+'/tree.dat', 'rb') as f:
			root = pickle.load(f)

		time = 0.05*snapnum/(2*np.pi)
		# Grab the 20 most massive bodies to plot
		snap = pb.load('data/'+sim+'/fullDiskVHi1.'+str(snapnum))
		pl = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
		snap = pb.load('data/'+sim+'/'+icfile)
		plic = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
		df1_d, df1_s = np.loadtxt('data/'+sim+'/delete1', dtype='int', unpack=True)
		pl['iord'] = prev_iorder(pl['iord'], len(plic), df1_d)
		massive_ind = np.argsort(np.array(pl['mass']))[::-1][:20]
		pl = pl[massive_ind]
		a_init = get_root_property(plic, pl, root, time, 'a')

		sort_a_ind = np.argsort(pl['a'])
		for idx, i in enumerate(sort_a_ind):
			pl_p = p_orbit(pl['a'][i])
			xvals.append(pl_p)
			child_p = p_orbit(a_init[i])
			yvals.append((pl_p-np.mean(child_p))/pl_p)
			svals.append(pl['mass'][i]/np.min(plic['mass']))


	xvals = []
	yvals = []
	svals = []
	get_data('fdvhia', 248000, 'fullDiskVHia.ic')
	get_data('fdvhib', 248000, 'fullDiskVHib.ic')
	get_data('fdvhic', 248000, 'fullDiskVHic.ic')
	get_data('fdvhid', 248000, 'fullDiskVHid.ic')
	get_data('fdvhie', 248000, 'fullDiskVHie.ic')
	axes.scatter(xvals, yvals, s=np.asarray(svals)*0.001, edgecolor='black', marker='o', label='fdHi')

	xvals = []
	yvals = []
	svals = []
	get_data('fdvhisteepa', 345000, 'fullDiskVHisteepa.ic')
	get_data('fdvhisteepb', 324000, 'fullDiskVHisteepb.ic')
	get_data('fdvhisteepc', 329000, 'fullDiskVHisteepc.ic')
	get_data('fdvhisteepd', 327000, 'fullDiskVHisteepd.ic')
	get_data('fdvhisteepe', 348000, 'fullDiskVHisteepe.ic')
	axes.scatter(xvals, yvals, s=np.asarray(svals)*0.001, edgecolor='black', marker='^', label='fdHiSteep')

	xvals = []
	yvals = []
	svals = []
	get_data('fdvhishallowa', 348000, 'fullDiskVHishallowa.ic')
	get_data('fdvhishallowb', 319000, 'fullDiskVHishallowb.ic')
	get_data('fdvhishallowc', 318000, 'fullDiskVHishallowc.ic')
	get_data('fdvhishallowd', 313000, 'fullDiskVHishallowd.ic')
	get_data('fdvhishallowe', 313000, 'fullDiskVHishallowe.ic')
	axes.scatter(xvals, yvals, s=np.asarray(svals)*0.001, edgecolor='black', marker='*', label='fdHiShallow')

	axes.legend()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_acc_zones_1():
	file_str = 'figures/acc_zones.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	with open('data/fdvhia/tree.dat', 'rb') as f:
		root = pickle.load(f)

	time = 0.05*248000/(2*np.pi)
	# Grab the 20 most massive bodies to plot
	snap = pb.load('data/fdvhia/fullDiskVHi1.248000')
	pl = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	snap = pb.load('data/fdvhia/fullDiskVHia.ic')
	plic = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	df1_d, df1_s = np.loadtxt('data/fdvhia/delete1', dtype='int', unpack=True)
	pl['iord'] = prev_iorder(pl['iord'], len(plic), df1_d)
	massive_ind = np.argsort(np.array(pl['mass']))[::-1][:20]
	pl = pl[massive_ind]
	a_init = get_root_property(plic, pl, root, time, 'a')

	fig, axes = plt.subplots(figsize=(16, 12))

	s = 1000
	amin, amax = 0.05, 0.2
	histbins = np.linspace(0, 100, 200)

	def p_orbit(a):
		return ((np.sqrt(a**3/mCentral))*u.yr).to(u.d).value

	def oli_plot(i, idx):
		print((pl['mass'][i]*u.M_sun).to(u.M_earth).value)
		m0 = (np.min(plic['mass'])*u.M_sun).to(u.M_earth).value
		print(len(a_init[i])*m0)
		print(len(a_init[i]))
		print('******')

		child_p = p_orbit(a_init[i])
		hist, bins = np.histogram(child_p, bins=histbins, density=True)
		bins = 0.5*(bins[1:] + bins[:-1])
		hist /= np.max(hist)/0.1
		axes.plot(bins, hist+0.2*idx, drawstyle='steps-mid', lw=0.5)
		axes.fill_between(bins, np.min(hist)+0.2*idx, hist+0.2*idx, alpha=0.5)
		pl_p = p_orbit(pl['a'][i])
		axes.vlines(pl_p, np.min(hist)+0.2*idx, np.max(hist)+0.2*idx, lw=lw)
		plt.gca().set_prop_cycle(None)
		axes.set_xlabel('Orbital Period [d]')
		axes.set_ylabel('Fraction of Planetesimals Accreted')
		axes.set_yticks([])
		#axes.set_xscale('log')

	# Sort by semimajor axis beore plotting
	sort_a_ind = np.argsort(pl['a'])

	for idx, i in enumerate(sort_a_ind):
		oli_plot(i, idx)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_acc_zones_old():
	file_str = 'figures/acc_zones.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(16, 12))

	s = 1000
	amin, amax = 0.05, 0.2
	histbins = np.linspace(0, 100, 200)

	root_idx = np.loadtxt('data/roots_vhi.txt')
	df1_d, df1_s = np.loadtxt('data/fullDiskVHi_delete1', dtype='int', unpack=True)
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
		axes.fill_between(bins, np.min(hist)+0.2*idx, hist+0.2*idx, color='gray')
		pl_p = p_orbit(plVHi['a'][massive_ind[i]])
		axes.vlines(pl_p, np.min(hist)+0.2*idx, np.max(hist)+0.2*idx, lw=lw)
		axes.set_xlabel('Orbital Period [d]')
		axes.set_ylabel('Fraction of Planetesimals Accreted')
		axes.set_yticks([])

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

	fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)

	axes = ax[0]
	btilde = 2*np.sqrt(3)

	surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHi['a'], p_vhi_ic['rbins'], surf_den)
	m_iso_vhi_at = np.sqrt((2*np.pi*(plVHi['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHi')

	surf_den = (p_vhif4_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	surf_den_at = np.interp(plVHif4['a'], p_vhif4_ic['rbins'], surf_den)
	m_iso_vhi_at = np.sqrt((2*np.pi*(plVHif4['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	axes.scatter(perf4.value, (plVHif4['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHif4')

	axes.axhline(1, ls='--', color='gray')
	axes.set_yscale('log')
	axes.set_ylim(8e-3, 20)
	axes.set_xlim(-5, 100)
	axes.set_ylabel(r'm / M$_{iso}$')
	axes.legend()

	axes = ax[1]
	f = 6
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha, lw=lw)

	f = 4
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha, lw=lw)

	f = 1
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, lw=lw)

	rhoval = (1*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
	r_pl_varyrho = (3*m_pl/(4*np.pi*rhoval))**(1/3)
	alpha1g = r_pl_varyrho/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, ls=':')

	rhoval = (5*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
	r_pl_varyrho = (3*m_pl/(4*np.pi*rhoval))**(1/3)
	alpha = r_pl_varyrho/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, ls=':')

	rhoval = (10*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
	r_pl_varyrho = (3*m_pl/(4*np.pi*rhoval))**(1/3)
	alpha10g = r_pl_varyrho/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, ls=':')

	axes.fill_between(prof_perIC, alpha1g, alpha10g, alpha=0.5)

	axes.axhline(0.1, ls='--', color='gray')
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'$\alpha$')

	fig.tight_layout()

	plt.subplots_adjust(wspace=0, hspace=0)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_f6f4_b():
	file_str = 'figures/f6f4_b.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)

	p1, p2 = 35, 60

	axes = ax[0]

	#surf_den = (p_vhi_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	#surf_den_at = np.interp(plVHi['a'], p_vhi_ic['rbins'], surf_den)
	#m_iso_vhi_at = np.sqrt((2*np.pi*(plVHi['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	#axes.scatter(per.value, (plVHi['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHi')

	axes.scatter(per.value, get_btilde(plVHi, p_vhi_ic), label='f = 6', edgecolor='black', linewidth=0.4)

	#surf_den = (p_vhif4_ic['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)
	#surf_den_at = np.interp(plVHif4['a'], p_vhif4_ic['rbins'], surf_den)
	#m_iso_vhi_at = np.sqrt((2*np.pi*(plVHif4['a']*u.AU).to(u.cm).value**2*btilde*surf_den_at)**3/(3*mCentralg))
	#axes.scatter(perf4.value, (plVHif4['mass']*u.M_sun).to(u.g).value/m_iso_vhi_at, label='fdHif4')

	axes.scatter(perf4.value, get_btilde(plVHif4, p_vhif4_ic), label='f = 4', edgecolor='black', linewidth=0.4)

	axes.axhline(2*np.sqrt(3), ls='--', color='gray')
	#axes.set_yscale('log')
	axes.set_ylim(0, 20)
	axes.set_xlim(-5, 100)
	axes.set_ylabel(r'Required $\tilde{b}$')
	axes.legend()

	axes.axvline(p1, ls='--', color=orange)
	axes.axvline(p2, ls='--', color=blue)

	axes = ax[1]
	f = 6
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha, lw=lw)

	f = 4
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	axes.semilogy(prof_perIC, alpha, lw=lw)

	f = 1
	alpha = r_pl*f/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, lw=lw)

	rhoval = (1*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
	r_pl_varyrho = (3*m_pl/(4*np.pi*rhoval))**(1/3)
	alpha1g = r_pl_varyrho/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, ls=':')

	rhoval = (5*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
	r_pl_varyrho = (3*m_pl/(4*np.pi*rhoval))**(1/3)
	alpha = r_pl_varyrho/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, ls=':')

	rhoval = (10*u.g/u.cm**3).to(u.M_sun/u.AU**3).value
	r_pl_varyrho = (3*m_pl/(4*np.pi*rhoval))**(1/3)
	alpha10g = r_pl_varyrho/(p_vhi_ic['a']*(m_pl/(3*mCentral))**(1/3))
	#axes.semilogy(prof_perIC, alpha, ls=':')

	axes.axvline(p1, ls='--', color=orange)
	axes.axvline(p2, ls='--', color=blue)

	axes.fill_between(prof_perIC.value, alpha1g, alpha10g, alpha=0.5)

	axes.axhline(0.1, ls='--', color='gray')
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel(r'$\alpha$')

	fig.tight_layout()

	plt.subplots_adjust(wspace=0, hspace=0)

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
	snap = pb.load('data/innerDiskLoFragLarge.5559000')
	plF = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perF = 2*np.pi*np.sqrt((plF['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

	fig, ax = plt.subplots(figsize=(8,8), nrows=2)
	axes = ax[0]
	axes.scatter(per0, pl0['e'], s=pl0['mass']/np.min(pl0['mass'])*0.1)
	axes.scatter(perNf, plNf['e'], s=plNf['mass']/np.min(pl0['mass'])*1, edgecolor='black', linewidth=0.4, label='Mergers Only')
	axes.scatter(perF, plF['e'], s=plF['mass']/np.min(pl0['mass'])*1, edgecolor='black', linewidth=0.4, label='Bounce-Merge')
	axes.set_yscale('log')
	axes.set_ylim(5e-5, 1e-1)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel('Eccentricity')
	axes.legend(loc=2)

	axes = ax[1]
	q1 = (plNf['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q1, bins=np.logspace(np.min(np.log10(q1)), np.max(np.log10(q1))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes.fill_between(bins, 0, hist, step='mid', alpha=0.5, color='orange')
	axes.plot(bins, hist, drawstyle='steps-mid', color='orange')
	axes.set_yticks([])

	q2 = (plF['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q2, bins=np.logspace(np.min(np.log10(q2)), np.max(np.log10(q2))))
	bins = 0.5*(bins[1:] + bins[:-1])
	axes.fill_between(bins, 0, hist, step='mid', alpha=0.5, color='green')
	axes.plot(bins, hist, drawstyle='steps-mid', color='green')
	axes.set_yticks([])
	axes.set_xscale('log')
	axes.set_yscale('log')

	axes.set_xlabel(r'Mass [M$_{\oplus}$]')
	axes.set_ylabel('dn/dm')

	stat, pval = stats.kstest(q1, q2)
	print('Frag ecc KS pval: '+ str(pval))

	# Cut out planetesimals
	mask1 = q1 > 1e-5
	mask2 = q2 > 1e-5
	stat, pval = stats.kstest(q1[mask1], q2[mask2])
	print('Frag ecc KS pval after cut: '+ str(pval))

	fig.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_frag_evo():
	file_str = 'figures/frag_evo.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, axes = plt.subplots(figsize=(8,8))

	axes.plot([], [])

	stepnumber, max_mass, mean_mass = np.loadtxt('data/innerDiskLoNoFrag.txt')
	axes.plot(stepnumber, max_mass/mean_mass, label='Mergers Only', lw=lw)

	stepnumber, max_mass, mean_mass = np.loadtxt('data/innerDiskLoFrag.txt')
	axes.plot(stepnumber, max_mass/mean_mass, label='Bounce + Mergers', lw=lw)

	axes.set_xlabel('Time Steps')
	axes.set_ylabel(r'M / $\langle$ m $\rangle$')
	axes.set_xscale('log')
	axes.legend()

	fig.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_frag_acc_zones():
	file_str = 'figures/frag_acc_zones.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8, 12), nrows=2)

	snap0 = pb.load('data/innerDiskLo.ic')
	pl0 = ko.orb_params(snap0, isHelio=True, mCentral=mCentral)

	s = 1000
	amin, amax = 0.05, 0.2
	histbins = np.linspace(0, 4, 100)


	# Grab the 3 most massive bodies to plot
	maxoli = 3

	def p_orbit(a):
		return ((np.sqrt(a**3/mCentral))*u.yr).to(u.d).value

	def oli_plot(i, idx, pl, axes, c):
		child_p = p_orbit(a_init[i])
		hist, bins = np.histogram(child_p, bins=histbins, normed=True)
		bins = 0.5*(bins[1:] + bins[:-1])
		hist /= np.max(hist)/0.1
		#axes.plot(bins, hist+0.2*idx, linestyle='steps-mid', color='gray', lw=0.5)
		axes.fill_between(bins, np.min(hist)+0.2*idx, hist+0.2*idx, color=c)
		pl_p = p_orbit(pl['a'][massive_ind[i]])
		axes.vlines(pl_p, np.min(hist)+0.2*idx, np.max(hist)+0.2*idx, lw=lw)
		axes.set_xlabel('Orbital Period [d]')
		axes.set_ylabel('Fraction of Planetesimals Accreted')
		axes.set_yticks([])
		print(pl['mass'][massive_ind[i]], len(child_p)*np.min(pl0['mass']), len(child_p))

	with open('data/innerDiskLoTree.dat', 'rb') as f:
		root = pickle.load(f)
	snap = pb.load('data/innerDiskLoNoFrag.6911000')
	plNf = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	time = 0.05*6911000/(2*np.pi)
	massive_ind = np.argsort(np.array(plNf['mass']))[::-1][:maxoli]
	pl = plNf[massive_ind]
	a_init = get_root_property(pl0, pl, root, time, 'a')
	# Sort by semimajor axis beore plotting
	indices = np.arange(0, maxoli)
	sort_a_ind = np.argsort(plNf['a'][massive_ind[indices]])
	for idx, i in enumerate(sort_a_ind):
		oli_plot(i, idx, plNf, ax[0], orange)

	with open('data/innerDiskLoFragLargeTree.dat', 'rb') as f:
		root = pickle.load(f)
	snap = pb.load('data/innerDiskLoFragLarge.5559000')
	plNf = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	time = 0.05*5559000/(2*np.pi)
	massive_ind = np.argsort(np.array(plNf['mass']))[::-1][:maxoli]
	pl = plNf[massive_ind]
	a_init = get_root_property(pl0, pl, root, time, 'a')
	# Sort by semimajor axis beore plotting
	indices = np.arange(0, maxoli)
	sort_a_ind = np.argsort(plNf['a'][massive_ind[indices]])
	for idx, i in enumerate(sort_a_ind):
		oli_plot(i, idx, plNf, ax[1], 'green')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def plot_rung_ecc():
	file_str = 'figures/rung_ecc.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	snap0 = pb.load('data/rungTest/rungTest.ic')
	pl0 = ko.orb_params(snap0, isHelio=True, mCentral=mCentral)
	per0 = 2*np.pi*np.sqrt((pl0['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/rungTest/fullDiskVHi1.054000')
	plNf = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perNf = 2*np.pi*np.sqrt((plNf['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)
	snap = pb.load('data/rungTestSingle/fullDiskVHi.216000')
	plF = ko.orb_params(snap, isHelio=True, mCentral=mCentral)
	perF = 2*np.pi*np.sqrt((plF['a']*u.AU).to(u.cm)**3/(G.cgs*mCentralg)).to(u.d)

	fig, ax = plt.subplots(figsize=(8,8), nrows=2)
	axes = ax[0]
	axes.scatter(per0, pl0['e'], s=pl0['mass']/np.min(pl0['mass'])*0.1)
	axes.scatter(perNf, plNf['e'], s=plNf['mass']/np.min(pl0['mass'])*0.1, edgecolor='black', linewidth=0.4, label='Variable Timesteps')
	axes.scatter(perF, plF['e'], s=plF['mass']/np.min(pl0['mass'])*0.1, edgecolor='black', linewidth=0.4, label='Fixed Timesteps')
	axes.set_yscale('log')
	axes.set_ylim(5e-5, 1e-1)
	axes.set_xlabel('Orbital Period [d]')
	axes.set_ylabel('Eccentricity')

	axes.legend(loc=4)

	axes = ax[1]

	q1 = (plNf['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q1, bins=np.logspace(np.min(np.log10(q1)), np.max(np.log10(q1))))
	bins = 0.5*(bins[1:] + bins[:-1])
	#axes.plot(bins, np.cumsum(hist), linewidth=lw)
	#axes.loglog(bins, hist, drawstyle='steps-mid')
	axes.fill_between(bins, 0, hist, step='mid', alpha=0.5, color='orange')
	axes.plot(bins, hist, drawstyle='steps-mid', color='orange')
	axes.set_yticks([])
	axes.set_yscale('log')

	q2 = (plF['mass']*u.M_sun).to(u.M_earth).value
	hist, bins = np.histogram(q2, bins=np.logspace(np.min(np.log10(q2)), np.max(np.log10(q2))))
	bins = 0.5*(bins[1:] + bins[:-1])
	#axes.plot(bins, np.cumsum(hist), linewidth=lw)
	#axes.loglog(bins, hist, drawstyle='steps-mid')
	axes.fill_between(bins, 0, hist, step='mid', alpha=0.5, color='green')
	axes.plot(bins, hist, drawstyle='steps-mid', color='green')
	axes.set_yticks([])
	axes.set_xscale('log')

	axes.set_xlabel(r'Mass [M$_{\oplus}$]')
	axes.set_ylabel('dn/dm')

	stat, pval = stats.kstest(q1, q2)
	print('Frag ecc KS pval: '+ str(pval))

	# Cut out planetesimals
	mask1 = q1 > 1e-5
	mask2 = q2 > 1e-5
	stat, pval = stats.kstest(q1[mask1], q2[mask2])
	print('Frag ecc KS pval after cut: '+ str(pval))

	fig.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

plot_timescales()
#plot_alpha_beta()
#plot_alpha_beta_evo()
#plot_alpha_beta_mass()
#plot_fulldisk_e_m_b()
#plot_alpha_pl_frac()
#plot_pl_frac_time()
#plot_surfden_profiles()
#plot_surfden_iso()
#plot_surfden_b()
#plot_smooth_acc()
#plot_acc_zones()
#plot_f6f4()
#plot_f6f4_b()
#plot_frag_ecc()
#plot_frag_evo()
#plot_frag_acc_zones()
#plot_rung_ecc()
