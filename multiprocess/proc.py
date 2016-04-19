import h5py
import numpy as np
from multiprocessing import Process, Pool

arr = None
stopAll = true

def f(ar):
	print id(arr)

if __name__ == '__main__':
	temp_arr = np.empty((1), np.float32)
    ctype = np.ctypeslib._typecodes[temp_arr.__array_interface__['typestr']]
    # create shared ctypes object with no lock
    shared_obj = mp.RawArray(ctype, 10)
    
    # create numpy array from shared object
    arr = np.frombuffer(shared_obj, dtype)
    arr = arr.reshape(shape)    

	p = Pool(processes=5)
	p.map_async(f, [0, 1, 2, 3, 4])
	p.close()
	p.join()

