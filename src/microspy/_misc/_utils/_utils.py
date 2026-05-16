#
# Copyright 2026 the microspy developer(s)
#
# This file is part of microspy.
#
# microspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# microspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with microspy. If not, see <http://www.gnu.org/licenses/>.
#
# This code is inspired by exspy : ~exspy._misc.elements.py
#

import numpy as np

def _get_table(
    data : np.ndarray,
    label : list | tuple
):
    """Structure data and labels to fit tabulate's functions
    
    Parameters
    ----------
    data
        Data to be printed
    label
        List of labels : will be printed at the left of each row 

    Returns
    -------
    table 
        List of lists that fits the tabulate functions

    Example
    -------
    >>> values_to_print = np.asarray(([1,2,3],[1,2,3],[1,2,3]))
    >>> values_to_print.shape
    (3, 3)
    >>> labels_to_print = ['row1','row2','row3']
    >>> _get_table(values_to_print, labels_to_print)
    array([['row1', 1, 2, 3],
           ['row2', 1, 2, 3],
           ['row3', 1, 2, 3]], dtype=object)
    """ 

    return np.insert(data.astype(object), 0, label, axis = 1)