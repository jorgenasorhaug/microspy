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
from tabulate import tabulate
from pathlib import Path
from ._utils import _path_exists

def _save_tabulated_data_as_csv(
    table : np.ndarray,
    headers : list | tuple,
    path : str = "",
    filename : str = "tabulate.csv",
    tbfmt : str = "%.3f"
):
    """Save tabulated data to csv format.

    Parameters
    ----------
    table
        Data to be printed. The data is expected to fit the shape 
        (len(header), len(label)).
    header
        List of headers : will be printed at the top of each column.
    path
        path to save the tabulated data. 
    filename
        Name of file.
    tbfmt
        Table format for values.
    """

    if _path_exists(path):

        tofile = str(
            Path(path) / Path(filename)
        )

        # Insert empty header columns:
        num = len(headers)
        diff = table.shape[-1] - num
        heads = "," * diff # Skipping diff num. cols

        # from list to single string
        for h in headers:
            heads += (h + ",")

        # If all headers are present:
        if diff == 0: num -= 1
            
        np.savetxt(
            fname = tofile,
            X = table,
            delimiter = ",",
            header = heads,
            fmt = "%s" + (("," + tbfmt ) * num)
        )

def _save_tabulated_data_as_txt(
    table : np.ndarray, 
    headers : list | tuple, 
    path : str = '', 
    filename : str = 'tabulate.txt',
):
    """Save the tabulated data to txt format.
    
    Parameters
    ----------
    table
        Data to be printed. The data is expected to fit the shape 
        (len(header), len(label)).
    header
        List of headers : will be printed at the top of each column.
    path
        path to save the tabulated data. 
    filename
        Name of file.
    """

    if _path_exists(path):

        tofile = str(
            Path(path) / Path(filename)
        )
    
        # Save the data as txt:
        with open(tofile, 'w') as f:  
            
            f.write(
                tabulate(
                    tabular_data = table,
                    headers = headers,
                    tablefmt="pretty"
                    )
                )

