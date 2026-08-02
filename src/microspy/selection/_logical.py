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

import numpy as np
from microspy.misc import exceptions

def logical_and(
    *args
) -> np.ndarray:
    """Identify particles fulfilling the intersection of the provided 
    properties (argument(s)). 

    Parameters
    ----------
    args
        Arguments passed on to :func:'numpy.logical_and.reduce'.
        Allowed types : list | tuple | ndarray.

    Returns
    -------
    mask
        Boolean ndarray of fulfilled conditions.
        
    Examples
    --------
    >>> import microspy as ms
    >>> s = ms.load(filename)
    >>> s.is_classified
    array([True, False, True, ..., False, False, True])
    >>> ms.classification.particles_logical_and(
            ~s.is_classified,
            s.Chemistry.data[:, s.elements.index("O")
        )
    array([False, False, False, ..., False, True, False])
    """
    
    for enum, arg in enumerate(args):
        if not isinstance(arg, np.ndarray | list | tuple):
            raise AttributeError(
                    f"Argument '{enum}' of type '{type(arg)}' is not "
                    "supported. Allowed argument types are ndarrays, lists "
                    "or tuples."
                )
        if len(arg) != len(args[0]):
            raise exceptions.ShapeError(
                    "All arguments must have identical shapes. "
                    f"Argument '{enum}' with shape ('"
                    f"{np.shape(args[enum])}') is different from argument 0's"
                    f"shape '{np.shape(args[0])}'."
                )
    
    return np.logical_and.reduce(args)

def logical_or(
    *args,
) -> np.ndarray:
    """Identify particles fulfilling the union of the provided properties 
    (argument(s)). 

    Parameters
    ----------
    args
        Arguments passed on to :func:'numpy.logical_or.reduce'.
    
    Returns
    -------
    mask
        Boolean array of fulfilled conditions.
        
    Examples
    --------
    See :func:'logical_and'.
    """
    
    for enum, arg in enumerate(args):
        if not isinstance(arg, np.ndarray | list | tuple):
            raise AttributeError(
                    f"Argument '{enum}' of type '{type(arg)}' is not "
                    "supported. Allowed argument types are ndarrays, lists "
                    "or tuples."
                )
        if len(arg) != len(args[0]):
            raise exceptions.ShapeError(
                    "All arguments must have identical shapes. "
                    f"Argument '{enum}' with shape ('"
                    f"{np.shape(args[enum])}') is different from argument 0's"
                    f"shape '{np.shape(args[0])}'."
                )
                
    return np.logical_or.reduce(args)
