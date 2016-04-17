import math
import numpy as np
import multiprocessing as mp

__all__ = [
            '_fix_slice',
            '_shape_after_slice',
            'empty_shared_array'
          ]

def _fix_slice(slc):
    """
    Fix up a slc object to be tuple of slices.
    slc = None is treated as no slc
    slc is container and each element is converted into a slice object
    None is treated as slice(None)
    
    Parameters
    ----------
    slc : None or sequence of tuples
        Range of values for slicing data in each axis.
        ((start_1, end_1, step_1), ... , (start_N, end_N, step_N))
        defines slicing parameters for each axis of the data matrix.  
    """
    if slc is None:
        return None # need arr shape to create slice
    fixed_slc = list()
    for s in slc:
        if not isinstance(s, slice):
            # create slice object
            if s is None or isinstance(s, int):
                # slice(None) is equivalent to np.s_[:]                
                # numpy will return an int when only an int is passed to np.s_[]
                s = slice(s)
            else:
                s = slice(*s)
        fixed_slc.append(s)
    return tuple(fixed_slc)


def _shape_after_slice(shape, slc):
    """
    Return the calculated shape of an array after it has been sliced.  
    Only handles basic slicing (not advanced slicing).
    
    Parameters
    ----------
    shape : tuple of ints
        Tuple of ints defining the ndarray shape
    slc : tuple of slices
        Object representing a slice on the array.  Should be one slice per
        dimension in shape.
    
    """
    if slc is None:
        return shape
    new_shape = list(shape)
    slc = _fix_slice(slc)
    for m, s in enumerate(slc):
        # indicies will perform wrapping and such for the shape
        start, stop, step = s.indices(shape[m])
        new_shape[m] = int(math.ceil((stop - start) / float(step)))
        if new_shape[m] < 0:
            new_shape[m] = 0
    return tuple(new_shape)


def empty_shared_array(shape, dtype=np.float32):
    # create a shared ndarray with the provided shape and type
    # get ctype from np dtype
    temp_arr = np.empty((1), dtype)
    ctype = np.ctypeslib._typecodes[temp_arr.__array_interface__['typestr']]
    # create shared ctypes object with no lock
    size = 1
    for dim in shape:
        size *= dim

    shared_obj = mp.RawArray(ctype, int(size))
    # create numpy array from shared object
    arr = np.frombuffer(shared_obj, dtype)
    arr = arr.reshape(shape)    

    return arr
