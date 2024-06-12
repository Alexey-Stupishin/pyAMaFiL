#!/usr/bin/env python3

import os, sys
import numpy as np
from mag_field_wrapper import MagFieldWrapper

#-------------------------------------------------------------------------------

m = sys.modules[__name__]
this_path = os.path.dirname(m.__file__)

lib_file = this_path + '/../src/AMaFiL/binaries/WWNLFFFReconstruction.dll'
print("Loading library from", lib_file)

maglib = MagFieldWrapper(lib_file)

# potential
print('Load potential cube ...')
maglib.load_cube(this_path + '/Data/11312_hmi.M_720s.20111010_085818.W120N23CR.CEA.POT.sav')
energy_pot = maglib.energy
print('Potential energy: ' + str(energy_pot) + ' erg')

# NLFFF
print('Load boundary cube ...')
maglib.load_cube(this_path + '/Data/11312_hmi.M_720s.20111010_085818.W120N23CR.CEA.BND.sav')
print('Calculate NLFFF ...')
box = maglib.NLFFF()
energy_nlfff = maglib.energy
print('NLFFF energy:     ' + str(energy_nlfff) + ' erg')

# Lines (seeds)
v = maglib.est_max_coords()

sz = maglib.get_box_size
input_seeds = np.zeros((np.prod(sz), 3), dtype = np.float64, order="C")

iz = 0
porosity = 10

cnt = 0
for iy in range(0, sz[1], porosity):
    for ix in range(0, sz[0], porosity):
        input_seeds[cnt, :] = [ix, iy, iz]
        cnt += 1

input_seeds = input_seeds[0:cnt, :]
print('Prepared ' + str(input_seeds.shape[0]) + ' seeds ...')

lines = maglib.lines(seeds = input_seeds, max_length = -1)
print('Passed ' + str(lines['n_passed']) + ', non-passed ' + str(lines['non_passed']))

# Lines (only passed)
lines = maglib.lines()
print('Passed ' + str(lines['n_passed']) + ', non-passed ' + str(lines['non_passed']))

# Lines (all)
lines = maglib.lines(reduce_passed = maglib.PASSED_NONE)
print('Passed ' + str(lines['n_passed']) + ', non-passed ' + str(lines['non_passed']))

pass
