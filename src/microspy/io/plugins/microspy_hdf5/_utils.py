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

def replace_none(obj : dict | list | tuple | None):
    """Replace None values from dictionaries, lists, typles, 
    or just None with empty strings"""
    if isinstance(obj, dict):
        return {k: replace_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_none(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(replace_none(v) for v in obj)
    elif obj is None: return ""
    else: return obj

def _string2listWithStrings(
    string : str,
    **kwargs
) -> list:
    """Convert a string to a list where each elements is separated by marker
    
    Parameters
    ----------
    string
        string to convert to list with elements
    **kwargs
        marker separating each elements. If None, "|" is used

    Returns
    -------
    List
        List of strings
    """
    if isinstance(string, list):
        return string

    marker = kwargs.get("marker") 
    if marker is None:
        marker = "|"
    
    List = string.split(marker)
    
    if len(List) == 1: List = List[0]

    return List

def _listWithStrings2string(
    List : list,
    **kwargs
) -> str:
    """Convert a list of strings to a single string where each elements is 
    separated by marker.

    Parameters
    ----------
    List
        List of strings
    **kwargs
        marker separating each elements. If None, "|" is used.

    Returns
    -------
    string
        string to convert to list with elements
    """
    if isinstance(List, str):
        return List

    marker = kwargs.get("marker") 
    if marker is None:
        marker = "|"
    
    string = ""
    for val in List:
        string += (marker + val)

    # Ignore the first inserted marker
    return string[1:]