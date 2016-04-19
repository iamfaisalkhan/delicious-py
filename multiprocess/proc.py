import h5py
import numpy as np
from multiprocessing import Process, Pool

arr = np.random.rand(10)

def f(ar):
	print id(arr)
	print arr[0]

if __name__ == '__main__':
	p = Pool(processes=5)
	p.map_async(f, [0, 1, 2, 3, 4])
	p.close()
	p.join()

