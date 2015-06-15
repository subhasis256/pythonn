import time
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import scikits.cuda.misc as scm
import pycuda.autoinit
import utils
import numpy as np

K = 87930
B = 1000

N = 1000

scm.init()

scores = gpuarray.to_gpu((5 * np.random.randn(B, K)).astype(np.float32))
probs = gpuarray.to_gpu(np.random.rand(B, K).astype(np.float32))

maxscores = gpuarray.empty((B,), dtype=np.float32)
maxscoreids = gpuarray.empty((B,), dtype=np.uint32)
deltas = gpuarray.empty_like(scores)
sumdeltas = gpuarray.empty((B,), dtype=np.float32)

cpu_probs = np.empty((B, K), dtype=np.float32)
indices = np.random.randint(0, K, size=(N, B)).astype(np.uint32)
gpu_ind = gpuarray.empty((B,), dtype=np.uint32)
selected_probs = gpuarray.empty((B,), dtype=np.float32)

for i in range(10):
    gpu_ind.set(indices[i])
    gpuarray.take(probs, gpu_ind, out=selected_probs)
    utils.scalar_sub(selected_probs, 1.0, selected_probs)
    gpuarray.multi_put([selected_probs], gpu_ind, out=[probs])

#print probs

t1 = time.clock()

for i in range(N):
    # get the softmax probs first
    utils.max(scores, 1, maxscores, maxscoreids)
    utils.sub_matvec(scores, maxscores, 0, deltas)
    cumath.exp(deltas, out=deltas)
    scm.sum(deltas, 1, sumdeltas)
    utils.div_matvec(deltas, sumdeltas, 0, probs)
#    probs.get(cpu_probs)
#    cpu_probs[np.arange(B), indices[i]] -= 1
#    probs.set(cpu_probs)
    gpu_ind.set(indices[i])
    gpuarray.take(probs, gpu_ind, out=selected_probs)
    utils.scalar_sub(selected_probs, 1.0, selected_probs)
    gpuarray.multi_put([selected_probs], gpu_ind, out=[probs])

t2 = time.clock()

#print probs

print 'tdiff = %.3f, per loop = %.6f, wps = %.3f' % ((t2-t1), (t2-t1)/N,
                                                     N*B/(t2-t1))
