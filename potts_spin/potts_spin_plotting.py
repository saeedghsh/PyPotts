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
import matplotlib.pyplot as plt
import numpy as np
import time

################################################################################
def plot_V_mat (V_log, KT, m,n,
                axis=None,
                skp=1, # skip in plotting V matrix to lighten the canvas
                save_to_file=False):
    ''''''
    ## if the axis is passed, just plot on it and return it, otherwise creat a new one
    return_axis = False if axis is None else True
    if axis is None: fig, axis = plt.subplots(1,1, figsize=(15,10))    

    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ## plot V matrix
    for row in range(rdim):
        for col in range(m,ndim):
            # axis.plot(np.arange(0,len(KT)+1,skp), V_log[::skp,row,col], '.-')
            axis.plot(np.arange(0,len(KT)+1,skp), V_log[::skp,row,col], ',-')

    ## set title
    axis.set_ylabel('V matrix')

    if return_axis:
        return axis

    else:
        if save_to_file:
            t = time.localtime()
            time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            fname = 'results/'+time_prefix+'_V_log.pdf'
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()


################################################################################
def plot_temperature (KT, axis=None, save_to_file=False):
    ''''''
    ## if the axis is passed, just plot on it and return it, otherwise creat a new one
    return_axis = False if axis is None else True
    if axis is None: fig, axis = plt.subplots(1,1, figsize=(15,10))    

    ## plot temprature
    axis.plot(KT)

    ## set title
    axis.set_ylabel('temprature')

    if return_axis:
        return axis

    else:
        if save_to_file:
            t = time.localtime()
            time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            fname = 'results/'+time_prefix+'_KT.pdf'
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

################################################################################
def plot_dsm_conversion_error (V_log, m,n,
                               axis=None,
                               save_to_file=False):
    ''''''
    ## if the axis is passed, just plot on it and return it, otherwise creat a new one
    return_axis = False if axis is None else True
    if axis is None: fig, axis = plt.subplots(1,1, figsize=(15,10))    

    ## dimensions of different matrices
    ndim, rdim = 2*m+n, m+n 

    ### plot normalization (doubly_stochastic) error
    col_err = [np.abs( np.ones(rdim) - V_log[itr,:rdim, m:].sum(axis=0) ).sum() for itr in range(V_log.shape[0])]
    row_err = [np.abs( np.ones(rdim) - V_log[itr,:rdim, m:].sum(axis=1) ).sum() for itr in range(V_log.shape[0])]
    axis.plot(col_err, 'r', label='column error')
    axis.plot(row_err, 'b', label='row error')

    axis.set_ylabel('normalization error\n (Doubly stochastic matrix)')
    axis.legend()

    ## 
    if return_axis:
        return axis

    else:
        if save_to_file:
            t = time.localtime()
            time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            fname = 'results/'+time_prefix+'_KT.pdf'
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()


################################################################################
def plot_V_KT_error (V_log, KT, m,n, config,
                     elapsed_time,
                     V_skp=1, # skip in plotting V matrix to lighten the canvas
                     save_to_file=False):
    '''
    '''
    ## constructing the figure
    fig, axes = plt.subplots(3,1, figsize=(18,10), sharex=True, sharey=False)

    ## set title
    ttl = 'vehicles:{:d} - tasks:{:d}'.format(m,n)
    ttl += ' -- {:d} iterations in {:.2f} seconds'.format(len(KT), elapsed_time)
    ttl += ' -- synchronous' if config['synchronous'] else ' -- asynchronous'
    ttl += '\n KT:(start:{:d}, step:{:.4f}, end:{:.4f})'.format(config['kT_start'], config['kT_step'], config['kT_end'])
    ttl += ' -- gamma:{:d}'.format(config['gamma'])
    ttl += ' -- normalization iteration:{:d}'.format(config['dsm_max_itr'])
    axes[0].set_title(ttl)

    ## plot temprature
    plot_temperature (KT, axis=axes[0], save_to_file=save_to_file)
   
    ## plot V matrix
    plot_V_mat (V_log, KT, m,n, axis=axes[1], skp=V_skp, save_to_file=save_to_file)

    ## plot normalization (doubly_stochastic) error
    plot_dsm_conversion_error (V_log, m,n, axis=axes[2], save_to_file=save_to_file)

    ## saving the figure
    if save_to_file:
        t = time.localtime()
        time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        fname = 'results/'+time_prefix+'.pdf'
        fig.savefig(fname, bbox_inches='tight')
        
    ## plot
    plt.tight_layout()
    plt.show()

