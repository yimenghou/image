
import h5py, pickle, gzip, numpy as np
import numpy as np


mode = "r"

if mode == "w":
	data = np.random.rand(1000, 1000)
	label = np.random.rand(1000, 1000)
	img_num = 10

	print data[:10]

	with h5py.File('example.h5','w') as f:
		f.create_dataset('dataset', data = data)
		f.create_dataset('labelset', data = label)
		f.create_dataset('n', data = img_num)

	with gzip.open("example.gz", "wb") as f:
		f.write(data)

	np.savetxt("example.txt", data)

	with file("example.pickle", "w") as f:
		pickle.dump(data, f)

elif mode == "r":

	'''
	h5 read
	'''
	with h5py.File("example.h5", "r") as f:
		data = f['dataset'][:]
		label = f['labelset'][:]
		n = f['n']

	'''
	gzip read
	'''
	# with gzip.open("example.gz", "rb") as f:
	# 	data = f.read()

	'''
	pickle read
	'''
	# with file("example.pickle", "r") as f:
	# 	pickle.dump(data, f)

	'''
	np read
	'''

	# np.savetxt("example.txt", data)




