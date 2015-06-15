from node import Node
from cpu_node import CPUNode
from gpu_node import GPUNode

import numpy as np

cpu = CPUNode("cpu0", 0)
x = cpu.allocate(None, np.float32, np.random.randn(5, 5))
y = cpu.allocate(None, np.float32, np.eye(5))

#print x
#print y

cpu.mul_update(x, y, out=y)
print y
