from node import Node
from cpu_node import CPUNode
from gpu_node import GPUNode

import numpy as np

import threading

gpu1 = GPUNode("gpu0", 0)
gpu2 = GPUNode("gpu1", 1)

x1 = gpu1.allocate(None, np.float32, np.random.randn(5, 5))
x2 = gpu2.allocate(None, np.float32, np.random.randn(5, 5))

y1 = gpu1.allocate(None, np.float32, np.eye(5))
y2 = gpu2.allocate(None, np.float32, np.eye(5))

print x1
print x2
print y1
print y2

gpu1.dot_update(x1, y1, out=y1, beta=1.0)
print y1

gpu2.mul_update(x2, y2, out=y2, beta=0.0)
print y2
