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

ALLOWED_VENDORS = [
    "Jeol"
]

def _image_directory_searcher(
    vendor : str
) -> str:
    """Search for image directory according to vendor by 
    walking through the sub-directories

    Parameters
    ----------
    vendor
        String stating the vendor

    Returns
    -------
    directory_searcher
        An image directory searcher
    """
    
    vendor = str(vendor)
    if vendor not in ALLOWED_VENDORS:
        raise ValueError(f"Vendor {vendor} not recognised "
                         "or supported yet.")
    
    """
    vendors ...
    readers ...
    """

    if vendor.lower() == "jeol":
        from .plugins.JEOL._utils import (
            search_for_image_directory as reader
        )

    return reader

