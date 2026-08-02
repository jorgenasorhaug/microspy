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
from numpy import dtype

def get_subdirectories(
    path : str
) -> list:
    """Returns a list of subdirectory names in the given path using 
    :func:'os.scandir'. 
    
    Note
    All files are excluded from the returned list.
    
    Parameters
    ----------
    path
        directory
        
    Returns
    -------
        list of subdirectories.
        
    Example
    -------
    >> import os
    >> get_subdirectories(os.getcwd())
    [dir1, dir2, dir3, ..., dirN]
    """
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]

def get_directory_filenames(path : str) -> list:
    """Returns a list of filenames within the given path using 
    :func:'os.scandir'.
    
    Note
    All subdirectories are excluded from the returned list.
    
    Parameters
    ----------
    path
        directory
        
    Returns
    -------
        list of subdirectories.
        
    Example
    -------
    >> import os
    >> get_subdirectories(os.getcwd())
    [file1, file2, file3, ..., fileN]
    """
    return [entry.name for entry in os.scandir(path) if not entry.is_dir()]

def _identify_subdirectories_of_interest(
    path : str,
    keyword : str
) -> tuple:
    """Identify all subdirectories containing the argument keyword.

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
    ('C:\\Users\\username', ['Documents']
    """

    parent_dir, child_dir = os.path.split(path)
    
    # If the keyword is already present in the parent directory:
    if keyword in child_dir:
        return parent_dir, [child_dir]

    parent_dir = path
    path = str(path)
    
    subdirs = get_subdirectories(path)

    child_dir = []
    for subdir in subdirs:
        if keyword in subdir: 
            child_dir.append(subdir)
    
    return parent_dir, child_dir

def _identify_filenames_of_interest(
    path : str,
    keyword : str
) -> list:
    """Identify all files containing the keyword.

    Parameters
    ----------
    path
        directory
    keyword
        keyword to search for
        
    Returns
    -------
    files   
        list of files containing keyword in their filename.

    Example
    -------
    >> _identify_subdirectories_of_interest(
        path = os.getcwd(),
        keyword = 'py'
    )
    ['particle_analysis.py', 'notebook.ipynb']
    """
    
    path = str(path)
    filenames = get_directory_filenames(path)

    get_names = []
    for name in filenames:
        if keyword in name: 
            get_names.append(name)
    
    return get_names

def _dtype_exists(data_type):
    """Check if data type exists.
    
    Parameters
    ----------
    data_type
        Data type
    
    Returns
    -------
    True if true, else False.
    """
    try:
        data_type = dtype(data_type)
        return True
    except TypeError:
        return False