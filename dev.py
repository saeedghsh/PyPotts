'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''

from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats, StringIO

import sys
if not(u'../' in sys.path): sys.path.append( u'../' )

from potts_spin import potts_spin as potts
from potts_spin import potts_spin_plotting as potplt
reload(potts)
reload(potplt)

np.set_printoptions(precision=2) # pricision of float print
np.set_printoptions(suppress=True) # to print in non-scientific mode

################################################################################
############################################################### Functions' Lobby
################################################################################


################################################################################
############################################################### Development Yard
################################################################################
'''
TODO:
> save V_log as txt file
> log E_local, E_task, E_loop and plot?
> how to animate the result (potts-pin-ann) in a meaningful way?
'''

########################################
###################### Setting parameter
########################################
config = {
    # coefficient of loop cost (E_loop)
    'gamma'                                 : 100,

    # number of iteration in conversion to doubly stochastic matric
    'dsm_max_itr'                           : 50,

    # termal setting
    'kT_start'                              : 100,
    'kT_step'                               : .9990,
    'kT_end'                                : .0010,

    # E_task computation mode
    # for detail see potts_spin.potts_spin.get_E_task()
    'E_task_computation_mode'               : ['summation', 'maximum'][1],

    # V update mode;
    # synchronous mode - the V matrix is updated as a whole.
    # asynchronous mode - the V matrix is updated one row or column at a time.
    'synchronous'                           : [False, True][0],

    # [asynchronous mode] row and columns could be updated;
    # randomly - row or column, and their indices are selected randomly
    # sequentially - first update columns one-by-one, then rows one-by-one
    'select_row_col_randomly'               : [False, True][0],

    # [asynchronous mode] [sequential updating] E_local could be updated;
    # once before updating each row or column
    # once for all rows and columns of a sequence (covering the whole matrix)
    'update_E_local_per_row_col'            : [False, True][1],

    # [asynchronous mode]
    # This flag specifies whether the temprature (kt) should change after each
    # row-col update.
    # If this flag is True, row-column are selected randomly, and E_local is
    # updated once before updating each row or column.
    # That is to say, if this flag is True, the two previous flags are ignored
    'update_temperature_after_each_row_col' : [False, True][1],

    # how often print the current iteration (does not print if set to 0)
    'verbose'                               : 500,
}

##### over-writting termal setting
config['kT_start'], config['kT_step'], config['kT_end'] = [ (100, .9990, .0100),
                                                            (100, .9995, .0100),
                                                            (100, .9990, .0010),
                                                            (100, .9990, .0001),
                                                            (100, .9980, .0010) ][3] #[0]
KT = [ config['kT_start'] ]
while KT[-1] > config['kT_end']: KT.append( KT[-1] * config['kT_step'] )
# print (len(KT))

########################################
################ Problem setting: Inputs
########################################
if 1:
    m, n = 2, 4 # vehicles, tasks
    D = np.loadtxt('sample_data/deltaMat_2_4.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
    T = np.array([0, 0, 438, 399, 487, 507, 0 , 0]) # time for [doing] each task
    
if 1:
    m, n = 3, 6 # vehicles, tasks
    D = np.loadtxt('sample_data/deltaMat_3_6.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
    T = np.array([0, 0, 0, 438, 599, 300, 421, 347, 557, 0, 0 , 0]) # time for [doing] each task

########################################
############################## Execution
########################################
print('process started...')
pr = cProfile.Profile()
pr.enable()

tic = time.time()
V_log, KT, itr = potts.main(D, T, m, n, config)
elapsed_time = time.time()-tic

pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(['tottime', 'cumulative', 'ncalls', 'filename'][0])
ps.print_stats()
if 0: print (s.getvalue())

########################################
######## Print and visualize the results
########################################
reload(potplt)
reload(potts)

thr = [.3, .4, .5][2] # threshold for converting V to binary

if 1: print('{:d} iterations (stopped at {:d}) in {:.2f} seconds'.format(len(KT), itr, elapsed_time))

if 0: print('last V matrix:')
if 0: print( V_log[itr,:,:] )

if 1: potplt.plot_V_KT_error (V_log, KT, m, n, config, elapsed_time, V_skp=100, save_figure=[False,True][0])
if 0: potplt.plot_assignment_3d( V_log[itr,:,:], m,n, V_thr=thr)

if 0: print ('validity of the solution: ', potts.is_V_mat_valid(V_log[itr,:,:], m,n, V_thr=thr) )
if 0: 
    assignments = potts.parse_V_to_assignment(V_log[itr,:,:],m,n, V_thr=thr)
    print (assignments)
    for a in potts.parse_assignments_to_string(assignments,m,n): print (a)

################################################################################
########################################################## Visualization Gallery
################################################################################

################################################################################
#################################################################### TESTNG AREA
################################################################################
# if np.any(np.isnan(V_log)): print('the first NAN in V_log appears in iteration: {:d}'.format(np.where(np.isnan(V_log))[0][0]))
# if np.any(np.isinf(V_log)): print('the first INF in V_log appears in iteration: {:d}'.format(np.where(np.isinf(V_log))[0][0]))
if 0:
    m, n = (2, 4)
    V = np.array(
        [[0.  , 0.  , 0.  , 0.00, 0.00, 0.99, 0.  , 0.  ],
         [0.  , 0.  , 0.94, 0.05, 0.00, 0.00, 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.06, 0.92],
         [0.  , 0.  , 0.05, 0.  , 0.  , 0.  , 0.93, 0.00],
         [0.  , 0.  , 0.  , 0.94, 0.  , 0.  , 0.00, 0.05],
         [0.  , 0.  , 0.  , 0.  , 0.99, 0.  , 0.  , 0.01],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])

if 0:
    m, n = (3, 6)
    V = np.array(
        [[0.  , 0.  , 0.  , 0.04, 0.33, 0.33, 0.  , 0.00, 0.33, 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.04, 0.33, 0.33, 0.  , 0.00, 0.33, 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.04, 0.33, 0.33, 0.  , 0.00, 0.33, 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.90, 0.  , 0.04, 0.00, 0.00],
         [0.  , 0.  , 0.  , 0.86, 0.  , 0.  , 0.  , 0.00, 0.  , 0.  , 0.07, 0.07],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.87, 0.06, 0.  , 0.  , 0.01, 0.01],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.49, 0.49],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.00, 0.  , 0.  , 0.95, 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.12, 0.02, 0.  , 0.  , 0.41, 0.41],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])

if 0:
    m, n = (3, 6)
    V = np.array(
        [[0.  , 0.  , 0.  , 0.  , 0.08, 0.  , 0.  , 0.  , 0.91, 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.75, 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.71, 0.00, 0.  , 0.  , 0.24, 0.06, 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.84, 0.00, 0.14, 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.02, 0.  , 0.99, 0.  , 0.  , 0.  , 0.  , 0.00, 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.85, 0.  , 0.  , 0.  , 0.13, 0.00],
         [0.  , 0.  , 0.  , 0.  , 0.06, 0.  , 0.  , 0.  , 0.  , 0.  , 0.86, 0.06],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.03, 0.  , 0.92],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.96, 0.  , 0.00],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])

if 0:
    m, n = (3, 6)
    V = np.array(
        [[0.  , 0.  , 0.  , 0.47, 0.19, 0.  , 0.  , 0.14, 0.2 , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.12, 0.11, 0.1 , 0.1 , 0.16, 0.41, 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.9 , 0.1 , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.7 , 0.3 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.4 , 0.  , 0.6 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.31, 0.  , 0.  , 0.68],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.08, 0.92, 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.59, 0.  , 0.08, 0.  , 0.32],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])

if 0: potplt.plot_assignment_3d( V, m,n )
