
import numpy as np
import time

tic = time.time()

a = np.random.randn(320, 50) + 1j*np.random.randn(320, 50)
U, s, V = np.linalg.svd(a, full_matrices=True)

toc = time.time()
U.shape, V.shape, s.shape

# S = np.zeros((320, 50), dtype=complex)
# S[:320, :50] = np.diag(s)
# np.allclose(a, np.dot(U, np.dot(S, V)))



#print U
print s
#print V
print "timeELAPSED: %.3f"%(toc-tic)
