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

from microspy.io._utils import (
    _identify_subdirectories_of_interest
)

def _estimate_number_of_particles_based_on_folders(
    path : str,
    folders : list,
    keyword = 'Particle'
) -> int:
    """Estimate the total number of analysed particles from the total number 
    of particle image folders.

    Parameters
    ----------
    path
        Folder directory.
    folders
        List of folder paths.
    keyword
        Look for keyword to estimate the number of particles.
        
    Returns
    -------
    """
    from pathlib import Path
    
    num_particles = 0
    
    path = Path(path)

    for fol in folders:

        _, p_folders = _identify_subdirectories_of_interest(
            path = path / fol,
            keyword = keyword
        )

        num_particles += len(p_folders)

    return num_particles

def search_for_image_directory(
    path : str,
    keyword : str = "Sutb",
    **kwargs
) -> str:
    """Search for potential sub-directory where images are stored. If 
    multiple subdirectories are found with the keyword, an error will be
    raised. 

    Parameters
    ----------
    path
        Where to start searching

    Returns
    -------
    directory
        Identified directory. Empty string if none is found. 
    """
    
    path = str(path)
    
    _dirs = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if keyword in name:
                _dirs.append(os.path.join(root, name))
    if len(_dirs) == 1: return _dirs[0]
    elif len(_dirs) == 0: return ""
    else: 
        # Look for Experiment ID in the keywrods argument:
        exp_ID = kwargs.get("experiment_folder_ID")
        if exp_ID < 10: exp_ID = f"0{exp_ID}"
        if exp_ID is not None:
            for _dir in _dirs:
                if f"{keyword}{exp_ID}" in _dir: return _dir
        
        # Raise Error if None were found:
        raise TypeError(
        f"{len(_dirs)} potential directories were found. "
        "Specify a directory to load correct images.")