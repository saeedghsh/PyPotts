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
import numpy as np

################################################################################
##################################################################### Potts Spin
################################################################################
def convert_to_doubly_stochastic (mat, max_itr=10, verbose=False):
    '''
    iterative algorithm by Sinkhorn and Knopp:
    alternately rescale all rows and all columns of A to sum to 1.
    '''
    for _ in range(max_itr):
        # vect = mat.sum(axis=1)
        # mat /= np.stack([ vect for _ in range(mat.shape[0])], axis=1)
        mat /= np.stack( [mat.sum(axis=1)] * mat.shape[0], axis=1)
                
        # vect = mat.sum(axis=0)
        # mat /= np.stack([ vect for _ in range(mat.shape[0])], axis=0)
        mat /= np.stack( [mat.sum(axis=0)] * mat.shape[0], axis=0)

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
    # Y = P.T / np.stack([np.diag(P) for _ in range(ndim)], axis=1)
    Y = P.T / np.stack( [np.diag(P)] * ndim, axis=1)
    E_loop = Y / np.where(np.abs(1-Y) < np.spacing(10), np.spacing(10), 1-Y )

    return E_loop

################################################################################
def get_E_task(D, T, m, n, config, V, P):
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

    if config['E_task_computation_mode'] == 'summation':
        # left_side = D + np.stack([L for _ in range(ndim)], axis=1)
        # left_side *= np.stack([P[:,rdim:].sum(axis=1) for _ in range(ndim)], axis=0)
        # right_side = D + np.stack([R for _ in range(ndim)], axis=1)
        # right_side *= np.stack([P[:m,:].sum(axis=0) for _ in range(ndim)], axis=1)
        left_side = D + np.stack( [L] * ndim, axis=1)
        left_side *= np.stack( [P[:,rdim:].sum(axis=1)] * ndim, axis=0)
        right_side = D + np.stack( [R] * ndim, axis=1)
        right_side *= np.stack( [P[:m,:].sum(axis=0)] * ndim, axis=1)
        
    elif config['E_task_computation_mode'] == 'maximum':
        # left_side = D + np.stack([L for _ in range(ndim)], axis=1)
        # x = rdim + np.argmax(L[rdim:])
        # left_side *= np.stack([P[:,x] for _ in range(ndim)], axis=0)
        # right_side = D + np.stack([R for _ in range(ndim)], axis=1)
        # y = np.argmax(R[:m])
        # right_side *= np.stack([P[y,:] for _ in range(ndim)], axis=1)

        left_side = D + np.stack( [L] * ndim, axis=1)
        x = rdim + np.argmax(L[rdim:])
        left_side *= np.stack( [P[:,x]] * ndim, axis=0)
        right_side = D + np.stack( [R] * ndim, axis=1)
        y = np.argmax(R[:m])
        right_side *= np.stack( [P[y,:]] * ndim, axis=1)
        
    else:
        raise ('E_task_computation_mode is unidentifiable')

    E_task = (left_side + right_side) / (2 * m * kappa)

    E_task[:,:m] = 1/np.spacing(1)
    E_task[rdim:,:] = 1/np.spacing(1)
    E_task[:m,rdim:] = 1/np.spacing(1)

    return E_task

################################################################################
def get_E_all(D, T, m, n, config, V):
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

    ## All Es
    E_loop = get_E_loop(P, m, n)
    E_task = get_E_task(D, T, m, n, config, V, P)
    E_local  = E_task + config['gamma'] * E_loop

    return E_local, E_task, E_loop

################################################################################
def update_V_synchronous (D, T, m, n, config, V, kt):
    '''
    updates all entries of the V matrix at the same time
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## get E_local
    E_local, _, _ = get_E_all(D, T, m, n, config, V)
    # E_local, E_task, E_loop = get_E_all(D, T, m, n, config, V)

    ## Update V Sync
    v_upd = np.exp( - E_local[:rdim, m:] / kt ) # crop the matrix (remove zeros)

    ## TODO: 
    ## I think the following "normalization" is redundant
    ## it is merely one step of the "convert_to_doubly_stochastic"
    # vect = v_upd.sum(axis=1) # denomintor for normalization
    # v_upd /= np.stack([ vect for _ in range(rdim)], axis=1) # normalize the V

    ## convert the V matrix to a doubly stochastic matrix
    v_dsm = convert_to_doubly_stochastic (v_upd, max_itr=config['dsm_max_itr'])

    ## reconstruct the original shape of the V, from (rdim,rdim) to (ndim, ndim)
    v_res = np.zeros((ndim, ndim))
    v_res[:rdim, m:] = v_dsm

    # return v_res, E_local, E_task, E_loop
    return v_res

################################################################################
def update_V_col (V, E_local, m, n, kt, col_idx, dsm_max_itr=20):
    ''''''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## update V - one column
    V[:rdim, col_idx] = np.exp( - E_local[:rdim, col_idx] / kt )

    ## convert the V matrix to a doubly stochastic matrix
    v_crp = V[:rdim, m:]
    v_dsm = convert_to_doubly_stochastic (v_crp, max_itr=dsm_max_itr)
    V = np.zeros((ndim, ndim))
    V[:rdim, m:] = v_dsm

    return V

################################################################################
def update_V_row (V, E_local, m, n, kt, row_idx, dsm_max_itr=20):
    ''''''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## update V - one row
    V[row_idx, m:] = np.exp( - E_local[row_idx, m:] / kt )

    ## convert the V matrix to a doubly stochastic matrix
    v_crp = V[:rdim, m:]
    v_dsm = convert_to_doubly_stochastic (v_crp, max_itr=dsm_max_itr)
    V = np.zeros((ndim, ndim))
    V[:rdim, m:] = v_dsm

    return V

################################################################################
def update_V_asynchronous_batch (D, T, m, n, config, V_in, kt):
    '''
    Updates the V matrix only one row/col at a time.  BUT: all rows
    and columns (the whole matrix) will be updated per each iteration,
    hence the name batch.  the is unlike the other approach to
    asynchronous version where a single row/column is updated per each
    iteration (in that mode, the selection of row and columns could be
    sequential or random)

    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    # need to copy, because V passed to this method is from [itr] and it should not
    # change, but an updated version should be returned to be stored in [itr+1][itr]
    V = V_in.copy()

    if not config['update_E_local_per_row_col']:
        E_local, _, _ = get_E_all(D, T, m, n, config, V)
        # E_local, E_task, E_loop = get_E_all(D, T, m, n, config, V)

    for col_idx in range(m,ndim):
        if config['update_E_local_per_row_col']:
            E_local, _, _ = get_E_all(D, T, m, n, config, V)
            # E_local, E_task, E_loop = get_E_all(D, T, m, n, config, V)
        if config['select_row_col_randomly']: col_idx = np.random.randint(m, ndim)
        V = update_V_col (V, E_local, m, n, kt, col_idx, config['dsm_max_itr'])

    for row_idx in range(0, rdim):
        if config['update_E_local_per_row_col']:
            E_local, _, _ = get_E_all(D, T, m, n, config, V)
            # E_local, E_task, E_loop = get_E_all(D, T, m, n, config, V)
        if config['select_row_col_randomly']: row_idx = np.random.randint(0, rdim)
        V = update_V_row (V, E_local, m, n, kt, row_idx, config['dsm_max_itr'])

    return V

################################################################################
def update_V_asynchronous_beta (D, T, m, n, config, V_log, KT):
    '''
    NOTE: that this beta version is not like the other asynchronous and
    synchronous. The others execute one iteration, this one runs all the
    iterations internally.

    Updates the V matrix only one row/col at a time

    updates E_local at each iteration
    selects row/col randomly
    updates kt after each row/col
    '''

    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    for itr, kt in enumerate(KT):
        if itr%500==0: print ('iteration: {:d}/{:d}'.format(itr, len(KT)))

        E_local, _, _ = get_E_all(D, T, m, n, config, V_log[itr, :, :])
        # E_local, E_task, E_loop = get_E_all(D, T, m, n, config, V_log[itr,:,:])
	        
        V = V_log[itr, :, :].copy()

	if config['select_row_col_randomly']:
	    if np.random.rand() > .5:
                col_idx = np.random.randint(m, ndim)
            	V_log[itr+1, :, :] = update_V_col (V, E_local, m, n, kt, col_idx, config['dsm_max_itr'])
            else:
            	row_idx = np.random.randint(0, rdim)
            	V_log[itr+1, :, :] = update_V_row (V, E_local, m, n, kt, row_idx, config['dsm_max_itr'])
            	
            	
    	else:
    	    # todo: this is new, double check
    	    rc_idx = itr % (2*rdim)
    	    if rc_idx < rdim:
    	    	col_idx = rc_idx+m
    	    	V_log[itr+1, :, :] = update_V_col (V, E_local, m, n, kt, col_idx, config['dsm_max_itr'])
    	    else:
    	    	row_idx = rc_idx - rdim
    	    	V_log[itr+1, :, :] = update_V_row (V, E_local, m, n, kt, row_idx, config['dsm_max_itr'])    	    	

        if np.any(np.isnan(V_log[itr+1,:,:])) or np.any(np.isinf(V_log[itr+1,:,:])):
            print('*** NOTE *** : process stopped at iteration {:d}, a NAN/INF appeared in V_log'.format(itr))
            break

    return V_log, itr

# m = 2
# n = 4
# ndim, rdim = 2*m+n, m+n
# for itr in range(40):
#     rc_idx = itr % (2*rdim)
#     if rc_idx < rdim:
# 	print ('column {:d}'.format(rc_idx+m))
#     else:
# 	print ('row {:d}'.format(rc_idx - rdim))

################################################################################
def main(D, T, m, n, config):
    '''
    Input:
    D: float 2darray (square 2xM+N) -- (delta) [transport] cost matrix
    T: float 1darray (2xM+N)        -- time for [doing] each task
    m: int scalar                   -- number of vehicles
    n: int scalar                   -- number of tasks


    Parameters:
    kT: float start, step , end     -- temperature of the system
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

    ## Execution
    if (not config['synchronous']) and config['update_temperature_after_each_row_col']:
        ## asynchronous - special case
        # - update temperature after each row_col
        # - randomly select row or column, and their indices
        # - update E_local once before updating each row or column
        print ('asynchronous mode - update temperature after each row or column')
        V_log, itr = update_V_asynchronous_beta (D, T, m, n, config, V_log, KT)

    else:
        ## synchronous and asynchronous
        update_V = update_V_synchronous if config['synchronous'] else update_V_asynchronous_batch
        print ('synchronous mode' if config['synchronous'] else 'asynchronous mode')

        ## the main loop
        for itr, kt in enumerate(KT):
            if (config['verbose'] and itr%config['verbose']==0):
                print ('iteration: {:d}/{:d}'.format(itr, len(KT)))

            V_log[itr+1,:,:] = update_V (D, T, m, n, config, V_log[itr,:,:], kt)

            if np.any(np.isnan(V_log[itr+1,:,:])) or np.any(np.isinf(V_log[itr+1,:,:])):
                print('*** NOTE *** : process stopped at iteration {:d}, a NAN/INF appeared in V_log'.format(itr))
                break

    return V_log, KT, itr





################################################################################
#################################### Parsers and Validity checks of the V Matrix
################################################################################
class InvalidVMatrixError(Exception):
    '''
    Custom Error defined for invalid V matrix
    '''
    def __init__(self, msg):
        self.msg = msg

################################################################################
def is_V_mat_valid(V, m,n, V_thr=0.5):
    '''
    V matrix should be square and binary

    TODO:
    this function does not check for the loops
    `parse_V_to_assignment` does...
    how to include that simply here?
    '''
    ndim, rdim = 2*m+n, m+n

    V_ = np.where(V>V_thr,1,0)

    problem = [
        # some start went directly to an end
        np.any( V_[:m,rdim:] ),

        # something went to an start
        np.any( V_[:,:m] ),

        # end went to something
        np.any( V_[rdim:,:] ),

        # something has zero output
        np.count_nonzero( np.count_nonzero(V_[:rdim,m:], axis=1)==0 ),

        # something has more that one output
        np.count_nonzero( np.count_nonzero(V_[:rdim,m:], axis=1)>1 ),

        # something has zero input
        np.count_nonzero( np.count_nonzero(V_[:rdim,m:], axis=0)==0 ),

        # something has more that one input
        np.count_nonzero( np.count_nonzero(V_[:rdim,m:], axis=0)>1 )
    ]

    return False if any(problem) else True

################################################################################
def parse_V_to_assignment(V, m,n, V_thr=0.5):
    '''
    V matrix should be square and binary
    '''
    ndim, rdim = 2*m+n, m+n

    # default validity checks of the V matrix
    if not is_V_mat_valid(V, m,n, V_thr):
        raise (InvalidVMatrixError('is_V_mat_valid() method flagged the V mat as invalid'))

    # binarization of the V matrix
    V_ = np.where(V>V_thr, 1, 0)

    # dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    # this is a list of assignments, where each assignment is a
    # sequence of indices
    assignments = []

    # Note that openlist is not limited to range(m), because we need
    # to check for loops that might start from a task
    openlist = range(rdim)
    while len(openlist) > 0:

        # start an assignment from the first element of the openlist
        # and remove that element from the openlist
        assignment = [ openlist.pop(0) ]

        # loop until the assignment sequence ends to an ending node
        while assignment[-1] < rdim:

            # find the next element that the last entry of the
            # assignment list points to.
            nxt = np.nonzero(V_[assignment[-1],:])[0][0]

            # safegaurd against loops
            if nxt in assignment:
                raise ( InvalidVMatrixError('loop detected') )
                # the break won't be reached. If you don't want to
                # raise an exception, and would like the method to
                # return assignments with loops, comment out the raise
                # and let it go to break
                break

            # add the next element to the sequence
            assignment.append( nxt )

            # when an assignment ends to an "ending node", do not try
            # to remove the nxt from the openlist, because it does is
            # not going to be included in the openlist and will raise
            # an error.
            if nxt >= rdim: break

            # remove the recently-added-to-assignment element from the
            # openlist
            openlist.pop( openlist.index(nxt) )

        # save the current assignment 
        assignments.append( assignment )

    # sagegaurding aginst assignment starting from a task
    if any([assignment[0] >= m for assignment in assignments]):
        raise (InvalidVMatrixError('assignment started from a task'))

    # Note: there is no safegaurd againt an assignment ending to a
    # starting node. That supposedly won't happen, since in that case
    # is_V_mat_valid() will return flag the V mat as invalid.

    return assignments

################################################################################
def idx_to_string(idx, m, n):
    ''''''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n
    
    if idx < m: # starting nodes
        return 'S{:d}'.format(idx+1)
    elif m <= idx < rdim: # task nodes
        return 't{:d}'.format(idx-m+1)
    elif rdim <= idx < ndim: # ending nodes
        return 'E{:d}'.format(idx-rdim+1)
    else:
        raise(IndexError('there is an assignment out of bound'))
        # return 'OutOfBoundIndex({:d}>={:d})'.format(idx,ndim)

################################################################################
def parse_assignments_to_string( assignments, m, n ):
    ''''''
    string = [' -> '.join([idx_to_string(idx, m, n) for idx in assignment])
               for assignment in assignments ]
    return string
