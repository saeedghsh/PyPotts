from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4) # pricision of float print
np.set_printoptions(suppress=True) # to print in non-scientific mode


################################################################################
############################################################### Functions' Lobby
################################################################################
def get_E_local(V, T, D, gamma, m, ndim, rdim):
    '''
    V: float 2darray (square 2xM+N) -- ordering of task
    T: float 1darray (2xM+N)        -- time for [doing] each task
    D: float 2darray (square 2xM+N) -- (delta) [transport] cost matrix

    gamma: int                      -- weighting factor

    m: int scalar                   -- number of vehicles
    n: int scalar                   -- number of tasks
    ndim: 2*m+n -- matrix size (V, D, ...)
    rdim: m+n --
    '''
    ## P
    P = np.linalg.inv( np.identity(ndim) - V) # P: measures the presence of loop in V

    ## E_loop
    Y = P.T / np.stack([np.diag(P) for _ in range(ndim)], axis=1)
    E_loop = Y / np.where(np.abs(1-Y) < np.spacing(10), np.spacing(10), 1-Y )

    ## E_task
    L = np.matmul( P.T , T+np.diag(np.matmul(V.T,D)) )
    R = np.matmul( P , T+np.diag(np.matmul(V,D.T)) )
    kappa = (L.max() + R.max()) *.5

    E_task = ( D + np.stack([L for _ in range(ndim)], axis=1) ) * np.stack([P[:,rdim:].sum(axis=1) for _ in range(ndim)], axis=0)
    E_task += ( D + np.stack([R for _ in range(ndim)], axis=1) ) * np.stack([P[:m,:].sum(axis=0) for _ in range(ndim)], axis=1) 
    E_task /= 2 * m * kappa

    E_task[:,:m] = 1/np.spacing(10)
    E_task[rdim:,:] = 1/np.spacing(10)
    E_task[:m,rdim:] = 1/np.spacing(10)

    ## E_local
    E_local  = E_task + gamma * E_loop

    return E_local


################################################################################
def convert_to_doubly_stochastic (mat, max_itr=10, verbose=False):
    '''
    iterative algorithm by Sinkhorn and Knopp:
    alternately rescale all rows and all columns of A to sum to 1.
    '''
    for _ in range(max_itr):
        mat /= np.stack([ mat.sum(axis=1) for _ in range(mat.shape[0])], axis=1)
        mat /= np.stack([ mat.sum(axis=0) for _ in range(mat.shape[0])], axis=0)
        
    if verbose:
        print('sum of cols:'), print(mat.sum(axis=0))
        print('sum of rows:'), print(mat.sum(axis=1))

    return mat
    
################################################################################
def update_V_synchronous (E_local, kt, m, rdim, ndim, dsm_max_itr=20):
    '''
    updates all entries of the V matrix at the same time
    '''
    ## Update V - Sync
    V_u = np.exp( - E_local / kt )[:rdim, m:]
    V_u /= np.stack([ V_u.sum(axis=1) for _ in range(rdim)], axis=1)

    ## convert the V matrix to a doubly stochastic matrix
    V_u = convert_to_doubly_stochastic (V_u, max_itr=dsm_max_itr)

    ## reconstruct the original shape of the V, from (rdim,rdim) to (ndim, ndim)
    V_r = np.zeros((ndim, ndim))
    V_r[:rdim, m:] = V_u
    
    return V_r

################################################################################
def update_V_asynchronous (E_local, kt, m, rdim, ndim, normalization_max_itr=20):
    '''
    Updates the V matrix only one row/col at a time
    '''
    ## Update V 
    V_u = np.exp( - E_local / kt )[:rdim, m:]
    V_u /= np.stack([ V_u.sum(axis=1) for _ in range(rdim)], axis=1)
    # print ('sum of columns:', V_u.sum(axis=0), '\n sum of rows:', V_u.sum(axis=1))
    
    ## convert the V matrix to a doubly stochastic matrix
    V_u = convert_to_doubly_stochastic (V_u, max_itr=dsm_max_itr)

    ## reconstruct the original shape of the V, from (rdim,rdim) to (ndim, ndim)
    V_r = np.zeros((ndim, ndim))
    V_r[:rdim, m:] = V_u
    
    return V_r


################################################################################
############################################################### Development Yard
################################################################################
'''
Input:
m: int scalar                   -- number of vehicles
n: int scalar                   -- number of tasks
D: float 2darray (square 2xM+N) -- (delta) [transport] cost matrix
T: float 1darray (2xM+N)        -- time for [doing] each task

Output:
V: float 2darray (square 2xM+N) -- ordering of task

Parameters:
kT: float start, step , end     -- temprature of the system


Process:
> init V -- random, normalized
> while t < thr
>> calculate E_local
>> update V (E_local)
>> normalize V
'''

##### parameters
gamma = 100 # coefficient of loop cost (E_loop)
dsm_itr_max = 50 # number of iteration in conversion to doubly stochastic matric

##### termal iteration
kT_start, kT_step, kT_end = [ (100, .9990, .0100),
                              (100, .9995, .0100),
                              (100, .9990, .0010),
                              (100, .9990, .0001),
                              (100, .9980, .0010) ] [4]

KT = [kT_start]
while KT[-1] > kT_end: KT.append( KT[-1] * kT_step )
print (len(KT))

##### Problem setting: Inputs
m, n = 2, 4 # vehicles, tasks
D = np.loadtxt('delta_mats/deltaMat_2_4.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
T = np.array([0, 0, 438, 599, 347, 557, 0 , 0]) # time for [doing] each task

m, n = 3, 6 # vehicles, tasks
D = np.loadtxt('delta_mats/deltaMat_3_6.txt', dtype=float, ndmin=2) # (delta) [transport] cost matrix
T = np.array([0, 0, 0, 438, 599, 300, 421, 347, 557, 0, 0 , 0]) # time for [doing] each task

ndim, rdim = 2*m+n, m+n # dimensions of different matrices

##### Creating a stack of V matrices
V_log = np.empty([len(KT)+1, ndim, ndim], dtype=float)

##### initializing the first V matrix
V_log[0,:,:] = np.ones((ndim,ndim), dtype=float) / rdim
V_log[0,   : , :m] = 0
V_log[0, -m: , : ] = 0

##### the main loop
print('process started...')

tic = time.time()
for itr, kt in enumerate(KT):
    if (itr % 500 == 0): print ('iteration: {:d}/{:d}'.format(itr, len(KT)))
    E_local = get_E_local(V_log[itr,:,:], T, D, gamma, m, ndim, rdim)
    V_log[itr+1,:,:] = update_V_synchronous (E_local, kt, m, rdim, ndim, dsm_max_itr=dsm_itr_max)

elapsed_time = time.time()-tic

##### print results
print('number of iterations: {:d}'.format( len(KT) ))
print('time elapsed: {:.2f}'.format(elapsed_time))
print('last V matrix:'), print( V_log[-1,:,:] )

################################################################################
########################################################## Visualization Gallery
################################################################################
### skip in plotting V matrix to lighten the canvas
skp = 100

### 
fig, axes = plt.subplots(3,1, figsize=(18,10), sharex=True, sharey=False)

### set title
ttl = 'vehicles:{:d} - tasks:{:d}'.format(m,n)
ttl += '\n KT:(start:{:d}, step:{:.4f}, end:{:.4f})'.format(kT_start,kT_step,kT_end)
ttl += '\n iterations:{:d}'.format(len(KT))
ttl += '\n gamma:{:d}'.format(gamma)
ttl += '\n normalization iteration:{:d}'.format(dsm_itr_max)
ttl += '\n time elapsed: {:.2f} seconds'.format(elapsed_time)
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
axes[2].set_ylabel('normalization error')
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
