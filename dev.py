from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats, StringIO

from potts_spin import potts_spin as potts
from potts_spin import potts_spin_plotting as potplt
# import sys
# if sys.version_info[0] == 3:
#     from importlib import reload
# elif sys.version_info[0] == 2:
#     pass
reload(potts)
reload(potplt)

np.set_printoptions(precision=4) # pricision of float print
np.set_printoptions(suppress=True) # to print in non-scientific mode

################################################################################
############################################################### Functions' Lobby
################################################################################


################################################################################
############################################################### Development Yard
################################################################################
'''
TODO:
> try variations of asynchronous update:
>>> do/don't update E_local at each iteration
>>> do/don't randomly select row/col
>>> do/don't update kt after each row/col

> log E_local, E_task, E_loop and plot?

> how to animate the result (potts-pin-ann) in a meaningful way?
'''

######################################## setting parameter
config = {'gamma': 100,  # coefficient of loop cost (E_loop)
          'dsm_max_itr': 50, # number of iteration in conversion to doubly stochastic matric
          'synchronous': [False, True][0], # V update mode - synchronous VS. asynchronous
      }

##### termal setting
config['kT_start'], config['kT_step'], config['kT_end'] = [ (100, .9990, .0100),
                                                            (100, .9995, .0100),
                                                            (100, .9990, .0010),
                                                            (100, .9990, .0001),
                                                            (100, .9980, .0010) ][4]

KT = [ config['kT_start'] ]
while KT[-1] > config['kT_end']: KT.append( KT[-1] * config['kT_step'] )
# print (len(KT))

# if not config['synchronous']: KT = KT[:100]

######################################## Problem setting: Inputs
if 1:
    m, n = 2, 4 # vehicles, tasks
    D = np.loadtxt('delta_mats/deltaMat_2_4.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
    T = np.array([0, 0, 438, 599, 347, 557, 0 , 0]) # time for [doing] each task
    
if 0:
    m, n = 3, 6 # vehicles, tasks
    D = np.loadtxt('delta_mats/deltaMat_3_6.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
    T = np.array([0, 0, 0, 438, 599, 300, 421, 347, 557, 0, 0 , 0]) # time for [doing] each task

######################################## execution
print('process started...')
pr = cProfile.Profile()
pr.enable()

tic = time.time()
V_log = potts.main(D, T, m, n, config, verbose=1)
elapsed_time = time.time()-tic

pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(['filename','cumulative'][0])
ps.print_stats()
# print (s.getvalue())

######################################## print results
print('{:d} iterations in {:.2f} seconds'.format(len(KT), elapsed_time))
# print('last V matrix:'), print( V_log[-1,:,:] )

################################################################################
############################################################# debugging workshop
################################################################################
# if np.any(np.isnan(V_log)): print('the first NAN in V_log appears in iteration: {:d}'.format(np.where(np.isnan(V_log))[0][0]))
# if np.any(np.isinf(V_log)): print('the first INF in V_log appears in iteration: {:d}'.format(np.where(np.isinf(V_log))[0][0]))

################################################################################
########################################################## Visualization Gallery
################################################################################
if 1:
    reload(potplt)
    potplt.plot_V_KT_error (V_log, KT, m, n, config, elapsed_time, V_skp=100, save_to_file=[False,True][1])

