from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats, StringIO

from potts_spin import potts_spin
# import sys
# if sys.version_info[0] == 3:
#     from importlib import reload
# elif sys.version_info[0] == 2:
#     pass
reload(potts_spin)


np.set_printoptions(precision=4) # pricision of float print
np.set_printoptions(suppress=True) # to print in non-scientific mode


################################################################################
############################################################### Functions' Lobby
################################################################################
# row_idx=None
# E_local,_,_ = potts_spin.get_E_local(V_log[-1,:,:], T, D, gamma, m, n)
# kt = KT[-1]

# ## dimensions of different matrices
# ndim, rdim = 2*m+n, m+n 

# ## if row index is not specified, pick a random row
# if row_idx is None: row_idx = np.random.randint(0, rdim)

# ## Update V - one row
# val = np.exp( - E_local[row_idx, m:] / kt ).sum()


# ## convert the V matrix to a doubly stochastic matrix
# v_dsm = convert_to_doubly_stochastic (v_upd, max_itr=dsm_max_itr)

# ## reconstruct the original shape of the V, from (rdim,rdim) to (ndim, ndim)
# v_res = np.zeros((ndim, ndim))
# v_res[:rdim, m:] = v_dsm




################################################################################
############################################################### Development Yard
################################################################################
'''
TODO:
> asynchronous update does not work

> try variations of asynchronous update:
>>> do/don't update E_local at each iteration
>>> do/don't randomly select row/col
>>> do/don't update kt after each row/col

> log E_local, E_task, E_loop and plot?

> how to animate the result (potts-pin-ann) in a meaningful way?

'''

##### parameters
gamma = 100 # coefficient of loop cost (E_loop)
dsm_max_itr = 50 # number of iteration in conversion to doubly stochastic matric

##### termal iteration
kT_start, kT_step, kT_end = [ (100, .9990, .0100),
                              (100, .9995, .0100),
                              (100, .9990, .0010),
                              (100, .9990, .0001),
                              (100, .9980, .0010) ] [4]

KT = [kT_start]
while KT[-1] > kT_end: KT.append( KT[-1] * kT_step )
# print (len(KT))

##### Problem setting: Inputs
if 1:
    m, n = 2, 4 # vehicles, tasks
    D = np.loadtxt('delta_mats/deltaMat_2_4.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
    T = np.array([0, 0, 438, 599, 347, 557, 0 , 0]) # time for [doing] each task
    
if 0:
    m, n = 3, 6 # vehicles, tasks
    D = np.loadtxt('delta_mats/deltaMat_3_6.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
    T = np.array([0, 0, 0, 438, 599, 300, 421, 347, 557, 0, 0 , 0]) # time for [doing] each task

ndim, rdim = 2*m+n, m+n

KT = KT[:500]
##### execution
print('process started...')
pr = cProfile.Profile()
pr.enable()

tic = time.time()
V_log = potts_spin.main(KT, T, D, gamma, m, n, dsm_max_itr, synchronous=1, verbose=1)
elapsed_time = time.time()-tic

pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(['filename','cumulative'][0])
ps.print_stats()

# print (s.getvalue())

##### print results
print('{:d} iterations in {:.2f} seconds'.format(len(KT), elapsed_time))
# print('last V matrix:'), print( V_log[-1,:,:] )

################################################################################
########################################################## Visualization Gallery
################################################################################
if 1:
    ### skip in plotting V matrix to lighten the canvas
    skp = 100
    
    ### 
    fig, axes = plt.subplots(3,1, figsize=(18,10), sharex=True, sharey=False)

    ### set title
    ttl = 'vehicles:{:d} - tasks:{:d}'.format(m,n)
    ttl += ' -- {:d} iterations in {:.2f} seconds'.format(len(KT), elapsed_time)
    ttl += '\n KT:(start:{:d}, step:{:.4f}, end:{:.4f})'.format(kT_start,kT_step,kT_end)
    ttl += ' -- gamma:{:d}'.format(gamma)
    ttl += ' -- normalization iteration:{:d}'.format(dsm_max_itr)

    axes[0].set_title(ttl)
    
    ### plot temprature
    axes[0].plot(KT)
    axes[0].set_ylabel('temprature')
    
    ### plot V matrix
    for row in range(rdim):
        for col in range(m,ndim):
            axes[1].plot(np.arange(0,len(KT)+1,skp), V_log[::skp,row,col], '.-')
    axes[1].set_ylabel('V matrix')

    ### plot normalization (doubly_stochastic) error
    col_err = [np.abs( np.ones(rdim) - V_log[itr,:rdim, m:].sum(axis=0) ).sum() for itr in range(V_log.shape[0])]
    row_err = [np.abs( np.ones(rdim) - V_log[itr,:rdim, m:].sum(axis=1) ).sum() for itr in range(V_log.shape[0])]
    axes[2].plot(col_err, 'r', label='column error')
    axes[2].plot(row_err, 'b', label='row error')
    axes[2].set_ylabel('normalization error\n (Doubly stochastic matrix)')
    axes[2].legend()

    ### drawing - saving
    if 0:
        ### save figure 
        t = time.localtime()
        time_prefix = '{:d}{:d}{:d}-{:d}{:d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        fname = 'results/'+time_prefix+'.pdf'
        fig.savefig(fname, bbox_inches='tight')
        # np.save('results/'+time_prefix+'_V_mat.npy', V_log)
        
        ### plot
        plt.tight_layout()
        plt.show()
        
    else:
        ### plot
        plt.tight_layout()
        plt.show()

