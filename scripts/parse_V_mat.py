'''Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi

This file is part of Arrangement Library. The of Arrangement Library
is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this program. If not, see
<http://www.gnu.org/licenses/>

'''

from __future__ import print_function
import time
import numpy as np

np.set_printoptions(precision=2) # pricision of float print
np.set_printoptions(suppress=True) # to print in non-scientific mode

import sys
if not(u'../' in sys.path): sys.path.append(u'../')

from potts_spin import potts_spin as potts
from potts_spin import potts_spin_plotting as potplt



################################################################################
if __name__ == '__main__':
    '''
    example
    -------
    python parse_V_mat.py --fname '../sample_data/3_6_Vmat__.txt' --m 3 --n 6 -verbose -visualize

    also,
    --bin_thr 0.5
    -save_figure
    '''

    args = sys.argv

    ###### fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]

    ###### fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break

    ##### 
    bin_thr = 0.5 if not('bin_thr' in locals()) else float(bin_thr)

    if not('fname' in locals()):
        raise(IOError('parameter missing: file name'))

    if not('m' in locals()):
        raise(IOError('parameter missing: number of vehicles (m)'))
    else:
        m = int(m)

    if not('n' in locals()):
        raise(IOError('parameter missing: number of tasks (n)'))
    else:
        n = int(n)

    ##### setting defaults values for visualization and saving options
    verbose = True if 'verbose' in options else False
    visualize = True if 'visualize' in options else False
    save_figure = True if 'save_figure' in options else False

    ################################################################################
    V = np.loadtxt(fname, dtype=float, delimiter=' ', ndmin=2, comments='#')

    if verbose:
        print ('\nNumber of Vehicles: {:d}\nNumber of tasks: {:d}'.format(m,n))

        print ('\nBinary V (thresholded at {:.2f}):'.format(bin_thr))
        print (np.where(V>bin_thr, 1 , 0) )

        print ('\nDoes V correspond to a valid solution? ', potts.is_V_mat_valid(V, m,n, V_thr=bin_thr) )
        
        print ('\nAssignments' )
        assignments = potts.parse_V_to_assignment(V ,m,n, V_thr=bin_thr)
        for a in potts.parse_assignments_to_string(assignments,m,n): print (a)
    
    if visualize or save_figure:
        potplt.plot_assignment_3d( V, m,n, V_thr=bin_thr, save_figure=save_figure)





