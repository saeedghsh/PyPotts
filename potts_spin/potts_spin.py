from __future__ import print_function
import numpy as np

################################################################################
def convert_to_doubly_stochastic (mat, max_itr=10, verbose=False):
    '''
    iterative algorithm by Sinkhorn and Knopp:
    alternately rescale all rows and all columns of A to sum to 1.
    '''
    for _ in range(max_itr):
        vect = mat.sum(axis=1)
        mat /= np.stack([ vect for _ in range(mat.shape[0])], axis=1)
        vect = mat.sum(axis=0)
        mat /= np.stack([ vect for _ in range(mat.shape[0])], axis=0)
        
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

    # with np.errstate(divide='raise'):
    #     try:
    #         a = np.abs(1-Y) < np.spacing(1)
    #     except FloatingPointError:
    #         print( np.abs(1-Y) )

    E_loop = Y / np.where(np.abs(1-Y) < np.spacing(10), np.spacing(10), 1-Y )

    return E_loop

################################################################################
def get_E_task(D, T, m, n, V, P):
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
def get_E_local(D, T, m, n, gamma, V):
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
    E_task = get_E_task(D, T, m, n, V, P)

    ## E_local
    E_local  = E_task + gamma * E_loop

    return E_local, E_task, E_loop
    
################################################################################
def update_V_synchronous (D, T, m, n, gamma, V, kt, dsm_max_itr=20):
    '''
    updates all entries of the V matrix at the same time
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## get E_local
    E_local, _, _ = get_E_local(D, T, m, n, gamma, V)

    ## Update V Sync 
    v_upd = np.exp( - E_local[:rdim, m:] / kt ) # crop the matrix (remove zeros)

    ## the following normalization is redundant
    ## it is merely one step of the "convert_to_doubly_stochastic"
    # vect = v_upd.sum(axis=1) # denomintor for normalization
    # v_upd /= np.stack([ vect for _ in range(rdim)], axis=1) # normalize the V

    ## convert the V matrix to a doubly stochastic matrix
    v_dsm = convert_to_doubly_stochastic (v_upd, max_itr=dsm_max_itr)

    ## reconstruct the original shape of the V, from (rdim,rdim) to (ndim, ndim)
    v_res = np.zeros((ndim, ndim))
    v_res[:rdim, m:] = v_dsm
    
    return v_res

################################################################################
def update_V_asynchronous (D, T, m, n, gamma, V, kt, dsm_max_itr=20):
    '''
    Updates the V matrix only one row/col at a time
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ##    
    E_local, _, _ = get_E_local(D, T, m, n, gamma, V)

    for col_idx in range(m,ndim):
        # E_local, _, _ = get_E_local(D, T, m, n, gamma, V)
        # if randomness: col_idx = np.random.randint(m, ndim)
        V[:rdim, col_idx] = np.exp( - E_local[:rdim, col_idx] / kt )
        
        ## convert the V matrix to a doubly stochastic matrix
        # print ('col_idx {:d} before dsm: '.format(col_idx), np.any(np.isnan(V)) )
        v_crp = V[:rdim, m:]
        v_dsm = convert_to_doubly_stochastic (v_crp, max_itr=dsm_max_itr)
        V = np.zeros((ndim, ndim))
        V[:rdim, m:] = v_dsm
        # print ('col_idx {:d} after dsm: '.format(col_idx), np.any(np.isnan(V)) )
        
    for row_idx in range(0, rdim):
        # E_local, _, _ = get_E_local(D, T, m, n, gamma, V)
        # if randomness: row_idx = np.random.randint(0, rdim)
        V[row_idx, m:] = np.exp( - E_local[row_idx, m:] / kt )

        ## convert the V matrix to a doubly stochastic matrix
        # print ('row_idx {:d} before dsm: '.format(row_idx), np.any(np.isnan(V)) )
        v_crp = V[:rdim, m:]
        v_dsm = convert_to_doubly_stochastic (v_crp, max_itr=dsm_max_itr)
        V = np.zeros((ndim, ndim))
        V[:rdim, m:] = v_dsm
        # print ('row_idx {:d} after dsm: '.format(row_idx), np.any(np.isnan(V)) )
    
    return V

################################################################################
def main(D, T, m, n, config, verbose=True):
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

    Output: V_log
    stack of V matrices, where;
    V: float 2darray (square 2xM+N) -- ordering of task
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 
    
    ## termal setting
    KT = [ config['kT_start'] ]
    while KT[-1] > config['kT_end']: KT.append( KT[-1] * config['kT_step'] )

    # if not config['synchronous']: KT = KT[:100]

    ## Creating a stack of V matrices
    V_log = np.empty([len(KT)+1, ndim, ndim], dtype=float)
    
    ## initializing the first V matrix
    V_log[0,:,:] = np.ones((ndim,ndim), dtype=float) / rdim
    V_log[0,   : , :m] = 0
    V_log[0, -m: , : ] = 0

    ## synchronous VS. asynchronous
    update_V = update_V_synchronous if config['synchronous'] else update_V_asynchronous
    print ('synchronous mode' if config['synchronous'] else 'asynchronous mode')

    ## the main loop
    for itr, kt in enumerate(KT):
        if (verbose and itr%500==0): print ('iteration: {:d}/{:d}'.format(itr, len(KT)))

        V_log[itr+1,:,:] = update_V (D, T, m, n, config['gamma'], V_log[itr,:,:], kt, config['dsm_max_itr'])

        if np.any(np.isnan(V_log[itr+1,:,:])) or np.any(np.isinf(V_log[itr+1,:,:])):
            print('*** NOTE *** : process stopped at iteration {:d}, a NAN/INF appeared in V_log'.format(itr))
            break

    return V_log
