import random, os

import h5py
import numpy as np
from multiprocessing import Process, Pool, RawArray, Queue

arr = None
dimX = None
dimY = None

def worker(q):
    print os.getpid()
    
    i = 0
    while i != -1:
        q.get()

    print np.sum(arr[np.s_[i:dimX:dimY]])

    print "done"

if __name__ == '__main__':

    f = h5py.File('/local/fkhan/data/tomopy/Hornby_ALS_2011.h5', "r")
    data = f["/exchange/data"]

    slices = 10
    temp_arr = np.empty((1), np.int32)
    ctype = np.ctypeslib._typecodes[temp_arr.__array_interface__['typestr']]
    # create shared ctypes object with no lock
    size = slices * data.shape[1] * data.shape[2]
    shared_obj = RawArray(ctype, size)
    
    # create numpy array from shared object
    arr = np.frombuffer(shared_obj, np.int32)
    arr = arr.reshape((slices, data.shape[1], data.shape[2]))

    dimX = data.shape[1]
    dimY = data.shape[2]

    sum = []
    

    NUM_OF_PROCESS = 5

    task_queue = Queue()

    p = None
    for i in range(NUM_OF_PROCESS):
        p = Process(target=worker, args=(task_queue, )).start()


    for i in range(slices):
        data.read_direct(arr, np.s_[i:dimX:dimY], np.s_[i:dimX:dimY])
        print "Read slice # %d"%i
        task_queue.put(i)


    for i in range(NUM_OF_PROCESS):
        task_queue.put(-1)

    p.join()
