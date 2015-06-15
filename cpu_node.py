"""
A CPU Node implementation with functions mostly from numpy
The DataType used is numpy ndarray
Note that this is *not* the threading implementation. This is merely to separate
the CPU functions from the GPU functions and provide an uniform abstraction
layer among them.
"""

from node import Node
import numpy as np

class CPUNode(Node):
    def __init__(self, name, cpu_id):
        """
        name: name of the node, can be any arbitrary string
        cpu_id: the integer id of the CPU that this Node should be running on
        """
        Node.__init__(self, name)
        self.cpu_id = cpu_id
        self.is_cpu = True
        self.is_gpu = False

    def allocate(self, shape, dtype, ary=None):
        if ary is not None:
            return ary.astype(dtype).copy()
        else:
            return np.empty(shape, dtype=dtype)

    def transfer_data_from(self, data, other_node):
        pass #TODO

    def _dot_update_impl(self, A, B, out, beta):
        # key point here: we can't simply say out = np.dot(A, B), since that
        # will simply create a *new* object out due to Python's passing
        # a copy of reference mechanism
        if beta == 0.0:
            np.dot(A, B, out=out)
        else:
            # why this hacky way? again, the answer is simply the fact that
            # Python's pass by reference copy means that you will ge a new
            # object if you simply assign to out
            out += (beta-1.0) * out + np.dot(A, B)

    def _mul_update_impl(self, x, y, out, beta):
        if beta == 0.0:
            np.multiply(x, y, out=out)
        else:
            out += (beta-1.0) * out + x * y

    def _add_update_impl(self, x, y, out, beta):
        if beta == 0.0:
            np.add(x, y, out=out)
        else:
            out += (beta-1.0) * out + x + y
