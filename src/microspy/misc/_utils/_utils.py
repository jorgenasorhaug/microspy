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

def _path_exists(
    path : str
) -> bool:
    """Check if directory exists. If not, the function will ask the user 
    whether to create it.

    Parameters
    ----------
    path
        directory
    """
    from pathlib import Path

    folder = Path(path)

    if not folder.exists():
        
        ans = input(f"Couldn't find folder {path}.\nCreate folder? (y/[n])")

        if ans.upper() == 'Y' or ans == '':

            print(f"Creating folder {path}")
            
            os.mkdir(path)

            return True

        else: return False

    else: return True