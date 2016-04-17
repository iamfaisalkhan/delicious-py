import h5py
import sys
import time
import numpy as np
import ctypes
import os
import math
import pyfftw

import multiprocessing as mp

# Helper utilities from tomopy code
from phase import _reciprocal_grid, _calc_pad, _paganin_filter_factor, _plan_effort
from util import *

shared_array = None

def read_hdf5(fname, grp, slc, shared=False, dtype=None):
    try:
        with h5py.File(fname, "r") as f:
            try:
                data = f[grp]
            except KeyError:
                # NOTE: I think it would be better to raise an exception here.
                print('Unrecognized hdf5 dataset: "%s"'%(str(grp)))
                return None
            shape = _shape_after_slice(data.shape, slc)
            
            if dtype is None:
                dtype = data.dtype

            if shared:
                arr = empty_shared_array(shape, dtype)
            else:
                arr = np.empty(shape, dtype)

            data.read_direct(arr, _fix_slice(slc))
    except KeyError:
        arr = None

    return arr

def read_exchange(fname, sino=None, prj=None):
    
    prj = read_hdf5(fname, "/exchange/data", (prj, sino), True, np.float32)
    flat = read_hdf5(fname, "/exchange/data_white", (None, sino), dtype=np.float32)
    dark = read_hdf5(fname, "/exchange/data_dark", (None, sino), dtype=np.float32)

    return prj, flat, dark

def normalize(params):
    # print "Started %d at %d" % (os.getpid(), time.time())

    global shared_array
    slc, axis, flat, dark = params
    arr =  slice_axis(shared_array, slc, axis)

    # Actual processing. 
    denom = flat - dark
    denom[denom < 1e-6] = 1e-6
    arr -= dark
    np.true_divide(arr, denom, arr)

def retrieve_phase(params):
    # print "Started retrieve_phase %d at %d" % (os.getpid(), time.time())
    global shared_array
    slc, phase_filter, px, py, prj, pad =  params
    tomo =  slice_axis(shared_array, slc, 0)

    dx, dy, dz = tomo.shape
    num_jobs = tomo.shape[0]
    for m in range(num_jobs):
        prj[px:dy + px, py:dz + py] = tomo[m]
        fproj = pyfftw.interfaces.numpy_fft.fft2(
            prj, planner_effort=_plan_effort(num_jobs))
        filtproj = np.multiply(phase_filter, fproj)
        proj = np.real(pyfftw.interfaces.numpy_fft.ifft2(
            filtproj, planner_effort=_plan_effort(num_jobs))
            ) / phase_filter.max()
        if pad:
            proj = proj[px:dy + px, py:dz + py]

        tomo[m] = proj

def slice_axis(arr, slc, axis):
    return arr[[slice(None) if i != axis else slc for i in range(arr.ndim)]]
 
def usage():
    print "Usage: %s %s"%(sys.argv[0], "filename.h5 dataset slice")

def main():
    
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    slc = (None, (1, 100))

    # Read the array
    t = time.time()
    prj, flat, dark = read_exchange(sys.argv[1], sino=slc[0], prj=slc[1])
    print "read file %s in %f sec" %(sys.argv[1], time.time() - t)

    global shared_array
    shared_array = prj

    # # Setup multi-process to compute frame sum. 
    ncores = 10
    prjs = prj.shape[0]
    nchunk = int(math.ceil(prjs / ncores))

    p = mp.Pool(processes=ncores)
    flat = flat.mean(axis=0)
    dark = dark.mean(axis=0)
    multiple_results = [ (np.s_[i:i+nchunk], 0, flat, dark) for i in range(0, prjs, nchunk) ]
    p.map_async(normalize, multiple_results)
    
    p.close()
    p.join()

    print np.sum(prj, axis=(2, 1))

    shared_array = prj

    py, pz, val = _calc_pad(prj, 1e-4, 50, 20, True)

    # Compute the reciprocal grid.
    dx, dy, dz = prj.shape
    w2 = _reciprocal_grid(1e-4, dy + 2 * py, dz + 2 * pz)

    # Filter in Fourier space.
    phase_filter = np.fft.fftshift(
        _paganin_filter_factor(20, 50, 1e-2, w2))

    # Enable cache for FFTW.
    pyfftw.interfaces.cache.enable()

    prj2 = val * np.ones((dy + 2 * py, dz + 2 * pz), dtype='float32')
    p = mp.Pool(processes=ncores)
    multiple_results = [ (np.s_[i:i+nchunk], phase_filter, py, pz, prj2, True) for i in range(0, prjs, nchunk) ]
    #p = mp.Pool(processes=ncores)
    #p.map_async(retrieve_phase, multiple_results)
    retrieve_phase(multiple_results[0])

    p.close()
    p.join()

    print np.sum(prj, axis=(2, 1))

# Read HDF5
if __name__ == "__main__":
   main()

