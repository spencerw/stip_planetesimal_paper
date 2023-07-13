=============================================================================
Wallace et al. (2023) data products from simulations of short-period
planetesimal accretion
README File Version 1.0
July, 2023
=============================================================================

-----

GENERAL INFORMATION

This is a brief README file for the data products associated with 
the Wallace et al. (2023) ApJ publication:

"Planetesimal Accretion at Short Orbital Periods"

Any queries/feedback about these data products should be sent to 
Spencer Wallace (University of Washington; scw7@uw.edu)

The simulation data, along with a python script to generate the
figures in the manuscript are provided in a single file.

-----

The simulation data itself is contained within the 'data' folder.
This folder contains four subdirectories:

	-annulus
	Contains the data products for all of the simulations run in
	a narrow annulus. This includes figures 3, 4, 5, 13 and 15.

	-mmsn
	Contains data products for all simulations run using a
	surface density slope of \alpha = 1.5. There are three
	subdirectories contained within.
		-f4 A simulation in which the artifical collision
		cross section was reduced. Used to construct figure 12.
		-hi The standard 'fdVHi' simulation referenced in the
		manuscript.
		-large The 'fdLo' simulation referenced in the
		manuscript.

	-shallow
	Contains data products for all simulations run using a
	surface density slope of \alpha = 0.5

	-steep
	Contains data products for all simulations run using a
	surface density slope of \alpha = 2.5

Note that there are five versions of the 'mmsn hi', 'shallow'
and 'steep' simulations. Each set is qualitatively identical
and the initial conditions were generated using a different
random number seed.

-----

The simulation snapshots are provided in TIPSY format and can
be read with the python package pynbody. For a demonstration
of how to do this, see the 'makeFigs.py' file.

The .ic files are the initial conditions in TIPSY format, while
the .iord files contain supplementary information to be read
by pynbody which stores the particle ID numbers.

The .txt files are secondary data products which contain measurements
of the maximum and average mass as a function of time for some of the
simulations. These are provided in cases where a large number of
timesteps were required and a high temporal resolution was needed.

The 'mmsn hi', 'shallow' and 'steep' simulations also contain a set of
.coll files and a 'delete' file. These contain the collision events and
IDs of the particles that were consumed during each merger event. This
information is used to construct figure 10. In addition, each simulation
has an associated 'tree.dat' file, which contains information about the
progenitors of each final body. See the 'plot_acc_zones' function in
'makeFigs.py' to see how to use the tree files.

-----

The 'makeFigs.py' script requires the following python packages, all of
which can be installed with pip:

- pynbody
- natsort
- KeplerOrbit