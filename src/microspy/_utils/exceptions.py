import numpy as np

class ShapeError(Exception):
    def __init__(self, message, errors = None):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
            
        # Now for your custom code...
        self.errors = errors

class FileNotFoundError(Exception):
    def __init(self, message, errors = None):
        super().__init__(message)
        self.errors = errors

def _check_for_numpy_ndarray(array):
    #Helping function to check if array is a numpy array
    return type(array) == np.ndarray

def _check_for_tuple_or_list(arg):
    """CHeck if the argument type is a list or a tuple."""
    arg_type = type(arg)
    if arg_type not in (list, tuple): raise TypeError(f"Input argument {arg} is not a list or a tuple.")
    else: return True

def _check_for_same_shape(array1, array2):
    """Helping function to check if two arrays are of the same shape"""
    if ~_check_for_numpy_ndarray(array1): raise TypeError(f'First argument of type {type(array1)} is not a numpy array')
    if ~_check_for_numpy_ndarray(array2): raise TypeError(f'Second argument of type {type(array2)} is not a numpy array')
    return array1.shape == array2.shape