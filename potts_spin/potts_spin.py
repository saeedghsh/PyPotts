from __future__ import print_function
import numpy as np

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
def get_E_loop(P, m, n):
    '''
    P: float 2darray (square 2xM+N) -- measures the presence of loop in V
    m: int scalar                   -- number of vehicles
    n: int scalar                   -- number of tasks
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ## E_loop
    Y = P.T / np.stack([np.diag(P) for _ in range(ndim)], axis=1)
    E_loop = Y / np.where(np.abs(1-Y) < np.spacing(1), np.spacing(1), 1-Y )

    return E_loop

################################################################################
def get_E_task(V, T, D, P, m, n):
    '''
    V: float 2darray (square 2xM+N) -- ordering of task
    T: float 1darray (2xM+N)        -- time for [doing] each task
    D: float 2darray (square 2xM+N) -- (delta) [transport] cost matrix
    P: float 2darray (square 2xM+N) -- measures the presence of loop in V

    m: int scalar                   -- number of vehicles
    n: int scalar                   -- number of tasks
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ## E_task
    L = np.matmul( P.T , T+np.diag(np.matmul(V.T,D)) )
    R = np.matmul( P , T+np.diag(np.matmul(V,D.T)) )
    kappa = (L.max() + R.max()) *.5

    E_task = ( D + np.stack([L for _ in range(ndim)], axis=1) ) * np.stack([P[:,rdim:].sum(axis=1) for _ in range(ndim)], axis=0)
    E_task += ( D + np.stack([R for _ in range(ndim)], axis=1) ) * np.stack([P[:m,:].sum(axis=0) for _ in range(ndim)], axis=1) 
    E_task /= 2 * m * kappa

    E_task[:,:m] = 1/np.spacing(1)
    E_task[rdim:,:] = 1/np.spacing(1)
    E_task[:m,rdim:] = 1/np.spacing(1)

    return E_task


################################################################################
def get_E_local(V, T, D, gamma, m, n):
    '''
    V: float 2darray (square 2xM+N) -- ordering of task
    T: float 1darray (2xM+N)        -- time for [doing] each task
    D: float 2darray (square 2xM+N) -- (delta) [transport] cost matrix

    gamma: int                      -- weighting factor

    m: int scalar                   -- number of vehicles
    n: int scalar                   -- number of tasks
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ## P
    P = np.linalg.inv( np.identity(ndim) - V) # P: measures the presence of loop in V

    ## E_loop
    E_loop = get_E_loop(P, m, n)

    ## E_task
    E_task = get_E_task(V, T, D, P, m, n)

    ## E_local
    E_local  = E_task + gamma * E_loop

    return E_local, E_task, E_loop
    
################################################################################
def update_V_synchronous (E_local, kt, m, n, dsm_max_itr=20):
    '''
    updates all entries of the V matrix at the same time
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

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
def update_V_asynchronous (E_local, kt, m, n, normalization_max_itr=20):
    '''
    Updates the V matrix only one row/col at a time
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ## Updated V 
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
def main(KT, T, D, gamma, m, n, dsm_max_itr, verbose=True):
    '''
    Input:
    m: int scalar                   -- number of vehicles
    n: int scalar                   -- number of tasks
    D: float 2darray (square 2xM+N) -- (delta) [transport] cost matrix
    T: float 1darray (2xM+N)        -- time for [doing] each task
    
    Parameters:
    kT: float start, step , end     -- temprature of the system
    gamma: int                      -- coefficient of loop cost (E_loop)
    dsm_max_itr = 50 # number of iteration in conversion to doubly stochastic matric

    Output:
    stack of V matrices, where;
    V: float 2darray (square 2xM+N) -- ordering of task
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 
    
    ## Creating a stack of V matrices
    V_log = np.empty([len(KT)+1, ndim, ndim], dtype=float)
    
    ## initializing the first V matrix
    V_log[0,:,:] = np.ones((ndim,ndim), dtype=float) / rdim
    V_log[0,   : , :m] = 0
    V_log[0, -m: , : ] = 0

    ## the main loop    
    for itr, kt in enumerate(KT):
        if (verbose and itr%500==0): print ('iteration: {:d}/{:d}'.format(itr, len(KT)))
        E_local, _, _ = get_E_local(V_log[itr,:,:], T, D, gamma, m, n)
        V_log[itr+1,:,:] = update_V_synchronous (E_local, kt, m, n, dsm_max_itr=dsm_max_itr)

    return V_log


