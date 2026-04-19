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

import os

def get_subdirectories(path : str):
    """
    Returns a list of immediate subdirectory names 
    in the given path using os.scandir().
    """
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]

def get_directory_filenames(path : str):
    """
    Returns a list of immediate subdirectory names in 
    the given path using os.scandir().
    """
    return [entry.name for entry in os.scandir(path) if not entry.is_dir()]

def _identify_subdirectories_of_interest(
    path : str,
    keyword : str
):
    """Identify all subdirectories starting with keyword 
    (argument)

    Parameters
    ----------
    path
        directory
    keyword
        keyword to search for
        
    Returns
    -------
    parent_dir
        Parent directory
    child_dir
        List of child directory(/-ies)

    Example
    -------
    >> _identify_subdirectories_of_interest(
        path = 'C:/Users/username/Documents',
        keyword = 'Doc'
    )
    ('C:\\Users\\username', ['Documents'])
    
    >> _identify_subdirectories_of_interest(
        path = 'C:/Users/username',
        keyword = 'Doc'
    )
    ('C:\\Users\\username', ['Documents'])
    """

    parent_dir, child_dir = os.path.split(path)
    
    # If the keyword is already present in the parent
    # directory, return it:
    if keyword in child_dir:

        return parent_dir, [child_dir]

    parent_dir = path
        
    path = str(path)
    
    subdirs = get_subdirectories(path)

    child_dir = []
    
    for subdir in subdirs:

        if keyword in subdir: child_dir.append(subdir)
    
    return parent_dir, child_dir

def _identify_filenames_of_interest(
    path : str,
    keyword : str
):
    """Identify all subdirectories starting with keyword 
    (argument)
    """
    path = str(path)
    
    filenames = get_directory_filenames(path)

    get_names = []
    
    for name in filenames:

        if keyword in name: get_names.append(name)
    
    return get_names