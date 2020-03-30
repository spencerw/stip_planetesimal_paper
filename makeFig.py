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

from scipy import stats

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

make_plot()