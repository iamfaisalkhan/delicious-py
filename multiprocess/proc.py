import random, os, time, sys

import h5py
import numpy as np
from multiprocessing import Process, Pool, RawArray, Queue

arr = None
dimX = None
dimY = None
task_queue = Queue()


def worker():
    pid = os.getpid()
    
    i = 0
    while i != -1:
        try:
            i = task_queue.get(False)
        except:
            time.sleep(0.5)
            continue

        if i != -1:
            print np.sum(arr[np.s_[i:dimX:dimY]])

    print "done"

if __name__ == '__main__':

    f = h5py.File('../../tomopy/Hornby_APS_2011.h5', "r")
    data = f["/exchange/data"]

    slices = 100
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

    
    NUM_OF_PROCESS = 5

    prcs = [Process(target=worker) for i in range(NUM_OF_PROCESS)]
    
    for prc in prcs:
        prc.start()

    for i in range(slices):
        data.read_direct(arr, np.s_[i:dimX:dimY], np.s_[i:dimX:dimY])
        print "Read slice # %d"%i
        task_queue.put(i)


    for i in range(NUM_OF_PROCESS):
        task_queue.put(-1)

    for prc in prcs:
        prc.join()
