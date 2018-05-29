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

from . import potts_spin as potts

################################################################################
############################################# plotting the history of iterations
################################################################################
################################################################################
################################################################################
def plot_V_mat (V_log, KT, m,n,
                axis=None,
                skp=1, # skip in plotting V matrix to lighten the canvas
                save_figure=False):
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
        if save_figure:
            t = time.localtime()
            time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            fname = 'results/'+time_prefix+'_V_log.pdf'
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)

        plt.tight_layout()
        plt.show()


################################################################################
def plot_temperature (KT, axis=None, save_figure=False):
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
        if save_figure:
            t = time.localtime()
            time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            fname = 'results/'+time_prefix+'_KT.pdf'
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)

        plt.tight_layout()
        plt.show()

################################################################################
def plot_dsm_conversion_error (V_log, m,n,
                               axis=None,
                               save_figure=False):
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
        if save_figure:
            t = time.localtime()
            time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            fname = 'results/'+time_prefix+'_KT.pdf'
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)

        plt.tight_layout()
        plt.show()


################################################################################
def plot_V_KT_error (V_log, KT, m,n, config,
                     elapsed_time,
                     V_skp=1, # skip in plotting V matrix to lighten the canvas
                     save_figure=False):
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
    fig.suptitle(ttl)
    # axes[0].set_title(ttl)

    ## plot temprature
    plot_temperature (KT, axis=axes[0], save_figure=save_figure)
   
    ## plot V matrix
    plot_V_mat (V_log, KT, m,n, axis=axes[1], skp=V_skp, save_figure=save_figure)

    ## plot normalization (doubly_stochastic) error
    plot_dsm_conversion_error (V_log, m,n, axis=axes[2], save_figure=save_figure)

    ## saving the figure
    if save_figure:
        t = time.localtime()
        time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        fname = 'results/'+time_prefix+'.pdf'
        fig.savefig(fname, bbox_inches='tight')
        
    ## plot
    plt.tight_layout()
    plt.show()


################################################################################
####################################### 3D plotting of the assignment (V matrix)
################################################################################
################################################################################
################################################################################
try:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
except:
    print ('3D vis is not available')    
    class FancyArrowPatch(): pass

############# https://gist.github.com/jpwspicer/ea6d20e4d8c54e9daabbc1daabbdc027
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

################################################################################
def draw_arrows(axis , X,Y,Z,V , r1,r2 , c1,c2,
                arrowstyle, arrowcolor, arrowwidth):
    ''''''
    ##### select starting and ending nodes
    rv, cv = np.meshgrid( range(r1,r2), range(c1,c2) )

    X_str = X[np.ravel(rv)]
    Y_str = Y[np.ravel(rv)]
    Z_str = Z[np.ravel(rv)]

    X_end = X[np.ravel(cv)]
    Y_end = Y[np.ravel(cv)]
    Z_end = Z[np.ravel(cv)]

    ##### 
    A_ = np.ravel( V[rv,cv] )

    ##### 
    for xs,ys,zs , xe,ye,ze , arrowalpha in zip(X_str, Y_str, Z_str, X_end, Y_end, Z_end, A_):
        a = Arrow3D( [xs,xe], [ys,ye], [zs,ze], mutation_scale=20, lw=arrowwidth,
                     alpha=arrowalpha, arrowstyle=arrowstyle, color=arrowcolor)
        axis.add_artist(a)

    return axis

################################################################################
def construct_nodes (m,n):
    ''''''
    ##### Vehicles starting nodes
    theta_v = np.linspace(0, 2*np.pi, num=m, endpoint=False)
    radiu_v = .5 * float(n) / m
    
    start_x = radiu_v * np.cos(theta_v)
    start_y = radiu_v * np.sin(theta_v)
    start_z = np.ones(m)
    
    ##### Vehicles ending nodes
    end_x = start_x.copy()
    end_y = start_y.copy()
    end_z = np.ones(m) * -1
    
    ##### task nodes
    theta_t = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    radiu_t = float(n) / m
    
    task_x = radiu_t * np.cos(theta_t)
    task_y = radiu_t * np.sin(theta_t)
    task_z = np.zeros(n)
    
    ##### nodes coordinates    
    X = np.concatenate([start_x, task_x, end_x])
    Y = np.concatenate([start_y, task_y, end_y])
    Z = np.concatenate([start_z, task_z, end_z])

    return X, Y, Z

################################################################################
def plot_assignment_3d_subplot(ax, V, m,n, arrowstyle='->',arrowcolor='k'):
    ''''''
    #####
    ndim, rdim = 2*m+n, m+n
    X, Y, Z = construct_nodes (m,n)

    ######################################## 3-layers rings
    alpha = 0.1
    t = np.linspace(0, 2*np.pi, num=60, endpoint=True)
    ## start
    r = .5 * float(n) / m
    ax.plot(r*np.cos(t), r*np.sin(t), np.ones(60), 'b-', alpha=alpha)
    ## end
    ax.plot(r*np.cos(t), r*np.sin(t), -1*np.ones(60), 'g-', alpha=alpha)
    ## task
    r = float(n) / m
    ax.plot(r*np.cos(t), r*np.sin(t), np.zeros(60), 'r-', alpha=alpha)

    ######################################## drawing nodes
    for idx in range(ndim):
        ax.text(X[idx], Y[idx], Z[idx], potts.idx_to_string(idx, m, n), color='k')

    ax.scatter(X[:m], Y[:m], Z[:m], c=['b']*m, s=r*50, marker='o', label='start')
    ax.scatter(X[m:rdim], Y[m:rdim], Z[m:rdim], s=r*50, c=['r']*n, marker='o', label='task')
    ax.scatter(X[rdim:], Y[rdim:], Z[rdim:], s=r*50, c=['g']*m, marker='o', label='end')

    ######################################## drawing edges
    ## start2task
    draw_arrows( ax, X,Y,Z,V, r1=0,r2=m , c1=m,c2=rdim, arrowstyle=arrowstyle, arrowcolor=arrowcolor, arrowwidth=r )
    ## task2task
    draw_arrows( ax, X,Y,Z,V, r1=m,r2=rdim , c1=m,c2=rdim, arrowstyle=arrowstyle, arrowcolor=arrowcolor, arrowwidth=r )
    ## task2end
    draw_arrows( ax, X,Y,Z,V, r1=m,r2=rdim , c1=rdim,c2=ndim, arrowstyle=arrowstyle, arrowcolor=arrowcolor, arrowwidth=r )

    ######################################## clean up and render
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_aspect('equal'), ax.legend()
    
    return ax

################################################################################
def plot_assignment_3d( V, m,n, V_thr=.5, arrowstyle='->',arrowcolor='k', save_figure=False):
    ''''''
    #####
    ndim, rdim = 2*m+n, m+n
    # X, Y, Z = construct_nodes (m,n)
    
    ##### create figure ans subplots
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # axis1 = fig.add_subplot(2,1,1, projection='3d')
    # axis2 = fig.add_subplot(2,1,2, projection='3d')
    fig, [axis1, axis2] = plt.subplots(1, 2, figsize=(18, 10), subplot_kw={'projection': '3d'})
    
    ##### subplot 1 - original V matrix
    axis1 = plot_assignment_3d_subplot(axis1, V, m,n, arrowstyle=arrowstyle, arrowcolor=arrowcolor)
    axis1.set_title('opacity of each link shows\n the wieght of the assignment')

    ##### subplot 2 - thresholded V matrix
    axis2 = plot_assignment_3d_subplot(axis2, np.where(V>V_thr,1,0), m,n, arrowstyle=arrowstyle, arrowcolor=arrowcolor)
    axis2.set_title('thresholded V matrix')

    ##### connecting the moving of the subplots
    # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once/41190487#41190487
    def on_move(event):
        ''''''
        # if event.inaxes == axis1:
        #     axis2.view_init(elev=axis1.elev, azim=axis1.azim)
        # elif event.inaxes == axis2:
        #     axis1.view_init(elev=axis2.elev, azim=axis2.azim)
        # else:
        #     return
        if event.inaxes == axis1:
            if axis1.button_pressed in axis1._rotate_btn:
                axis2.view_init(elev=axis1.elev, azim=axis1.azim)
            elif axis1.button_pressed in axis1._zoom_btn:
                axis2.set_xlim3d(axis1.get_xlim3d())
                axis2.set_ylim3d(axis1.get_ylim3d())
                axis2.set_zlim3d(axis1.get_zlim3d())
        elif event.inaxes == axis2:
            if axis2.button_pressed in axis2._rotate_btn:
                axis1.view_init(elev=axis2.elev, azim=axis2.azim)
            elif axis2.button_pressed in axis2._zoom_btn:
                axis1.set_xlim3d(axis2.get_xlim3d())
                axis1.set_ylim3d(axis2.get_ylim3d())
                axis1.set_zlim3d(axis2.get_zlim3d())
        else:
            return

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    ##### render
    if save_figure:
        t = time.localtime()
        time_prefix = '{:d}{:02d}{:02d}-{:02d}{:02d}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        fname = 'results/'+time_prefix+'_V_mat_assignment_in_3D.pdf'
        fig.savefig(fname, bbox_inches='tight')
        # plt.close(fig)

    plt.tight_layout() # fig.tight_layout()
    plt.show()
