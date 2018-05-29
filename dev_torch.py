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
np.set_printoptions(precision=2) # pricision of float print
np.set_printoptions(linewidth=150) # The number of characters per line for inserting line breaks (default = 80).
np.set_printoptions(suppress=True) # to print in non-scientific mode

import torch
torch.set_printoptions(precision=2) # pricision of float print
torch.set_printoptions(linewidth=150) # The number of characters per line for inserting line breaks (default = 80).

import matplotlib.pyplot as plt
# import cProfile, pstats, StringIO

# import sys
# if not(u'../' in sys.path): sys.path.append( u'../' )

from potts_spin import potts_spin as potts
from potts_spin import potts_spin_torch as potts_torch
from potts_spin import potts_spin_plotting as potplt

reload(potts)
reload(potts_torch)
reload(potplt)

################################################################################
############################################################### Functions' Lobby
################################################################################

################################################################################
############################################################### Development Yard
################################################################################

########################################
###################### Setting parameter
########################################
import method_config
reload(method_config)
config = method_config.config
# print(config['gamma'])

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

D = torch.from_numpy(D).type(torch.float)
T = torch.from_numpy(T).type(torch.float)
    
########################################
############################## Execution
########################################
print('process started...')
tic = time.time()

if 0: # numpy version
    if isinstance(D, np.ndarray):
        V_log, KT, itr = potts.main(D, T, m, n, config)
    elif isinstance(D, torch.Tensor):
        V_log, KT, itr = potts.main(D.numpy(), T.numpy(), m, n, config)

else: # pytorch version 
    if isinstance(D, torch.Tensor):
        V_log, KT, itr = potts_torch.main(D, T, m, n, config)
    elif isinstance(D, np.ndarray):
        V_log, KT, itr = potts_torch.main(torch.from_numpy(D).type(torch.float),
                                          torch.from_numpy(T).type(torch.float),
                                          m, n, config)
        
elapsed_time = time.time()-tic
print ('elapsed time:{:.2f}'.format(elapsed_time))

########################################
######## Print and visualize the results
########################################
thr = .5 

if isinstance(V_log, torch.Tensor): V_log = V_log.numpy()


if 1: print('{:d} iterations (stopped at {:d}) in {:.2f} seconds'.format(len(KT), itr, elapsed_time))

if 1: potplt.plot_V_KT_error (V_log, KT, m, n, config, elapsed_time, V_skp=100, save_figure=[False,True][0])

if 0: potplt.plot_assignment_3d( V_log[itr,:,:], m,n, V_thr=thr)


################################################################################
########################################################## Visualization Gallery
################################################################################

################################################################################
#################################################################### TESTNG AREA
################################################################################
