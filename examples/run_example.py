#!/usr/bin/env python3

import os, sys
import numpy as np
from MagFieldWrapper import MagFieldWrapper
from pathlib import Path

#-------------------------------------------------------------------------------

m = sys.modules[__name__]
this_path = os.path.dirname(m.__file__)

# print("Loading library from", lib_path)

# maglib = MagFieldWrapper(lib_path)
maglib = MagFieldWrapper(this_path + '../../../binaries/WWNLFFFReconstruction.dll')

print('Load potential cube ...')
# maglib.load_cube(data_path / '11312_hmi.M_720s.20111010_085818.W120N23CR.CEA.POT.sav')
maglib.load_cube('g:/BIGData/Work/ISSI/Work/Disambig/NoSmooth2/Trim/Potential/HMI+SST_combined_BOX.sav')
energy_pot = maglib.energy
print('Potential energy: ' + str(energy_pot) + ' erg')

v = maglib.est_max_coords()

sz = maglib.get_box_size
input_seeds = np.zeros((np.prod(sz), 3), dtype = np.float64, order="C")

iz = 1
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

lines = maglib.lines()
print('Passed ' + str(lines['n_passed']) + ', non-passed ' + str(lines['non_passed']))

pass
