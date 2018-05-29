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

########################################
###################### Setting parameter
########################################
config = {
    # coefficient of loop cost (E_loop)
    'gamma'                                 : 100,

    # number of iteration in conversion to doubly stochastic matric
    'dsm_max_itr'                           : 50,

    # termal setting
    'kT_start'                              : 100,
    'kT_step'                               : .9990,
    'kT_end'                                : .0010,

    # E_task computation mode
    # for detail see potts_spin.potts_spin.get_E_task()
    'E_task_computation_mode'               : ['summation', 'maximum'][0],

    # V update mode;
    # synchronous mode - the V matrix is updated as a whole.
    # asynchronous mode - the V matrix is updated one row or column at a time.
    'synchronous'                           : [False, True][0],

    # [asynchronous mode] slow versus fast cooling
    # This flag specifies whether the temprature (kt) should change after each
    # row-col update.
    # If this flag is True, row-column are selected randomly, and E_local is
    # updated once before updating each row or column.
    # That is to say, if this flag is True, the two next flags are ignored
    'update_temperature_after_each_row_col' : [False, True][1],

    # [asynchronous mode] [slow cooling] random versus sequentially
    # row and columns could be updated;
    # randomly - row or column, and their indices are selected randomly
    # sequentially - first update columns one-by-one, then rows one-by-one
    'select_row_col_randomly'               : [False, True][0],

    # [asynchronous mode] [slow cooling] E_local could be updated;
    # once before updating each row or column
    # once for all rows and columns of a sequence (covering the whole matrix)
    'update_E_local_per_row_col'            : [False, True][1],

    # how often print the current iteration (does not print if set to 0)
    'verbose'                               : 500,
}

##### over-writting termal setting
config['kT_start'], config['kT_step'], config['kT_end'] = [ (100, .9990, .0100),
                                                            (100, .9995, .0100),
                                                            (100, .9990, .0010),
                                                            (100, .9990, .0001),
                                                            (100, .9980, .0010) ][2] #[0]
