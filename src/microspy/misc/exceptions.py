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
import warnings

class ShapeError(Exception):
    """Raise a 'Shape error'."""
    def __init__(self, message, errors = None):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
            
        # Now for your custom code...
        self.errors = errors

class FileNotFoundError(Exception):
    """Raise a 'File not found error'."""
    def __init(self, message, errors = None):
        super().__init__(message)
        self.errors = errors
        
class InputError(Exception):
    """Raise an 'input error'."""
    def __init(self, message, errors = None):
        super().__init__(message)
        self.errors = errors
        
def formatted_warning(
    message : str, 
    *args,
    **kwargs
):
    """Raise a nicely formatted warning message.
    
    Parameters
    ----------
    message
        message to raise
    category
        Warning category. See builtins warning types.
    """
    original_formatwarning = warnings.formatwarning
    # Clean formatting:
    warnings.formatwarning = lambda msg, *args, **kwargs: f"⚠ {msg} ⚠\n"
    warnings.warn(message, UserWarning)
    # Reset formatwarning:
    warnings.formatwarning = original_formatwarning