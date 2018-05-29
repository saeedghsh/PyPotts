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
import torch

################################################################################
##################################################################### Potts Spin
################################################################################
def convert_to_doubly_stochastic (mat, max_itr=10, verbose=False):
    ''''''
    for _ in range(max_itr):
        mat /= torch.stack( [mat.sum(dim=1)] * mat.shape[0], dim=1)
        mat /= torch.stack( [mat.sum(dim=0)] * mat.shape[0], dim=0)
    return mat

################################################################################
def get_E_loop(P, m, n):
    ndim, rdim = 2*m+n, m+n

    Y = P.t() / torch.stack([torch.diag(P)] *ndim, dim=1)
    E_loop = Y / torch.where(torch.abs(1-Y) < np.spacing(10), torch.tensor(np.spacing(10)), 1-Y )

    return E_loop

################################################################################
def get_E_task(D, T, m, n, config, V, P):
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## E_task
    # print ( P.dtype, T.dtype, V.dtype, D.dtype )
    L = torch.matmul( P.t() , T+torch.diag(torch.matmul(V.t(),D)) )
    R = torch.matmul( P , T+torch.diag(torch.matmul(V,D.t())) )
    kappa = (torch.max(L) + torch.max(R)) *.5

    if config['E_task_computation_mode'] == 'summation':
        left_side = D + torch.stack([L] *ndim, dim=1)
        left_side *= torch.stack([P[:,rdim:].sum(dim=1)] *ndim, dim=0)
        right_side = D + torch.stack([R] *ndim, dim=1)
        right_side *= torch.stack([P[:m,:].sum(dim=0)] *ndim, dim=1)

    elif config['E_task_computation_mode'] == 'maximum':
        left_side = D + torch.stack([L] *ndim, dim=1)
        x = rdim + torch.argmax(L[rdim:])
        left_side *= torch.stack([P[:,x]] *ndim, dim=0)

        right_side = D + torch.stack([R] *ndim, dim=1)
        y = torch.argmax(R[:m])
        right_side *= torch.stack([P[y,:]] *ndim, dim=1)

    else:
        raise ('E_task_computation_mode is unidentifiable')

    E_task = (left_side + right_side) / (2 * m * kappa)

    E_task[:,:m] = torch.tensor( 1/np.spacing(10) )
    E_task[rdim:,:] = torch.tensor( 1/np.spacing(10) )
    E_task[:m,rdim:] = torch.tensor( 1/np.spacing(10) )

    return E_task

################################################################################
def get_E_all(D, T, m, n, config, V):
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n
    P = torch.inverse( torch.eye(ndim) - V)
    E_loop = get_E_loop(P, m, n)
    E_task = get_E_task(D, T, m, n, config, V, P)
    E_local  = E_task + config['gamma'] * E_loop

    return E_local, E_task, E_loop


################################################################################
def update_V_synchronous (D, T, m, n, config, V, kt):
    ''''''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## get E_local
    E_local, _, _ = get_E_all(D, T, m, n, config, V)

    ## Update V Sync
    v_upd = torch.exp( - E_local[:rdim, m:] / kt ) # crop the matrix (remove zeros)

    ## convert the V matrix to a doubly stochastic matrix
    v_dsm = convert_to_doubly_stochastic (v_upd, max_itr=config['dsm_max_itr'])

    ## reconstruct the original shape of the V, from (rdim,rdim) to (ndim, ndim)
    v_res = torch.zeros(ndim, ndim)
    v_res[:rdim, m:] = v_dsm

    return v_res


################################################################################
def update_V_col (V, E_local, m, n, kt, col_idx, dsm_max_itr=20):
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## update V - one column
    V[:rdim, col_idx] = torch.exp( - E_local[:rdim, col_idx] / kt )

    ## convert the V matrix to a doubly stochastic matrix
    v_crp = V[:rdim, m:]
    v_dsm = convert_to_doubly_stochastic (v_crp, max_itr=dsm_max_itr)
    V = torch.zeros(ndim, ndim)
    V[:rdim, m:] = v_dsm
    
    return V

################################################################################
def update_V_row (V, E_local, m, n, kt, row_idx, dsm_max_itr=20):
    ''''''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## update V - one row
    V[row_idx, m:] = torch.exp( - E_local[row_idx, m:] / kt )

    ## convert the V matrix to a doubly stochastic matrix
    v_crp = V[:rdim, m:]
    v_dsm = convert_to_doubly_stochastic (v_crp, max_itr=dsm_max_itr)
    V = torch.zeros((ndim, ndim))
    V[:rdim, m:] = v_dsm
    
    return V

################################################################################
def update_V_asynchronous_batch (D, T, m, n, config, V_in, kt):
    ''''''
        ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    # need to copy, because V passed to this method is from [itr] and it should not
    # change, but an updated version should be returned to be stored in [itr+1][itr]
    V = V_in.clone()

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
    ''''''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    for itr, kt in enumerate(KT):
        if itr%500==0: print ('iteration: {:d}/{:d}'.format(itr, len(KT)))

        E_local, _, _ = get_E_all(D, T, m, n, config, V_log[itr, :, :])
        # E_local, E_task, E_loop = get_E_all(D, T, m, n, config, V_log[itr,:,:])
            
        V = V_log[itr, :, :].clone()

        if np.random.rand() > .5:
            col_idx = np.random.randint(m, ndim)
            V_log[itr+1, :, :] = update_V_col (V, E_local, m, n, kt, col_idx, config['dsm_max_itr'])
        else:
            row_idx = np.random.randint(0, rdim)
            V_log[itr+1, :, :] = update_V_row (V, E_local, m, n, kt, row_idx, config['dsm_max_itr'])

        if np.isnan(V_log[itr+1,:,:]).any() or np.isinf(V_log[itr+1,:,:]).any():
            print('*** NOTE *** : process stopped at iteration {:d}, a NAN/INF appeared in V_log'.format(itr))
            break

    return V_log, itr

################################################################################
def main(D, T, m, n, config):
    '''
    '''
    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n

    ## termal setting
    KT = [ config['kT_start'] ]
    while KT[-1] > config['kT_end']: KT.append( KT[-1] * config['kT_step'] )

    ## Creating a stack of V matrices
    V_log = torch.empty(len(KT)+1, ndim, ndim, dtype=torch.float)

    ## initializing the first V matrix
    V_log[0,:,:] = torch.ones(ndim,ndim, dtype=torch.float) / rdim
    V_log[0,   : , :m] = 0
    V_log[0, -m: , : ] = 0

    
    # ## Execution
    # if config['synchronous']:
    #     ## synchronous and asynchronous
    #     update_V = update_V_synchronous

    #     ## the main loop
    #     for itr, kt in enumerate(KT):
    #         if (config['verbose'] and itr%config['verbose']==0):
    #             print ('iteration: {:d}/{:d}'.format(itr, len(KT)))

    #         V_log[itr+1,:,:] = update_V (D, T, m, n, config, V_log[itr,:,:], kt)

    #         if np.isnan(V_log[itr+1,:,:]).any() or np.isinf(V_log[itr+1,:,:]).any():
    #             print('*** NOTE *** : process stopped at iteration {:d}, a NAN/INF appeared in V_log'.format(itr))
    #             break

    # else:
    #     raise ( StandardError(' only synchronous :D ') )

    # return V_log, KT, itr


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

            if np.isnan(V_log[itr+1,:,:]).any() or np.isinf(V_log[itr+1,:,:]).any():
                print('*** NOTE *** : process stopped at iteration {:d}, a NAN/INF appeared in V_log'.format(itr))
                break
            
    return V_log, KT, itr


