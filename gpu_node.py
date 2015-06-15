"""
A GPU Node implemetation with functions from pycuda and scikits.cuda.
Note that each such instance *must* be called from a separate thread, since CUDA
only allows one handle per thread
"""

from node import Node
import scikits.cuda.misc as scm
import scikits.cuda.linalg as scl
import scikits.cuda.cublas as cublas
import pycuda.gpuarray as gpuarray
import pycuda.driver as driver

import utils

class GPUNode(Node):
    def __init__(self, name, gpu_id):
        """
        name: name of the node, can be any arbitrary string
        gpu_id: the integer id of the GPU that this Node should be running on
        """
        Node.__init__(self, name)
        self.ctx = driver.Device(gpu_id).make_context()
        self.device = self.ctx.get_device()
        print 'Executing on device at PCI ID:', self.device.pci_bus_id()
        self.handle = cublas.cublasCreate()
        self.gpu_id = gpu_id
        self.is_cpu = False
        self.is_gpu = True

    def __del__(self):
        """
        Cleanup pycuda stack
        """
        driver.Context.pop()

    def allocate(self, shape, dtype, ary=None):
        """
        allocate given shape pycuda.gpuarray on the device being managed by this
        Node object. If ary is not none, its content should be transferred to
        the gpu instead and the shape and dtype parameters should be ignored
        """
        if ary is not None:
            return gpuarray.to_gpu(ary.astype(dtype))
        else:
            return gpuarray.empty(shape, dtype=dtype)

    def transfer_data_from(self, data, other_node):
        pass #TODO

    def _dot_update_impl(self, A, B, out, beta):
        """
        ok we have got to do some work here since the scikits.cuda
        implementation is woefully incomplete for matrix-vector multiplies
        """
        # we do not have to check for beta == 0.0 here, since if beta == 0.0,
        # the implementation of cublasSgemm or cublasSgemv guarantees that out
        # doesnt have to be a valid array in that case
        utils.add_dot(A, B, out, beta=beta, handle=self.handle)

    def _mul_update_impl(self, x, y, out, beta):
        """
        use a custom built Elementwise kernel specifically created for
        pre-allocated memories
        """
        utils.mul(x, y, out=out, beta=beta)

    def _add_update_impl(self, x, y, out, beta):
        """
        use a custom built Elementwise kernel specifically created for
        pre-allocated memories
        """
        utils.add(x, y, out=out, beta=beta)

