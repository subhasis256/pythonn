import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel

import scikits.cuda.linalg as scl
import scikits.cuda.misc as scm
import scikits.cuda.cublas

import numpy as np

def add_vdot(M, v, out, beta=0.0, transM='N', handle=None):
    if handle is None:
        handle = scm._global_cublas_handle

    assert M.strides[1] <= M.strides[0], 'only C-order arrays supported'

    transM = transM.lower()
    if transM == 'n':
        trans = 't'
        m = M.shape[1]
        n = M.shape[0]
        alpha = 1.0
        lda = M.strides[0] // M.dtype.itemsize
        if v.shape[0] != M.shape[1] or out.shape[0] != M.shape[0]:
            raise ValueError('dimension mismatch: %s %s %s' %
                             (M.shape, v.shape, out.shape))
    elif transM == 't':
        trans = 'n'
        m = M.shape[1]
        n = M.shape[0]
        alpha = 1.0
        lda = M.strides[0] // M.dtype.itemsize
        if v.shape[0] != M.shape[0] or out.shape[0] != M.shape[1]:
            raise ValueError('dimension mismatch: %s %s %s' %
                             (M.shape, v.shape, out.shape))
    else:
        raise ValueError('transM must be n or t')

    if (M.dtype == np.complex64 and v.dtype == np.complex64):
        cublas_func = scikits.cuda.cublas.cublasCgemv
        alpha = np.complex64(alpha)
        beta = np.complex64(beta)
    elif (M.dtype == np.float32 and v.dtype == np.float32):
        cublas_func = scikits.cuda.cublas.cublasSgemv
        alpha = np.float32(alpha)
        beta = np.float32(beta)
    elif (M.dtype == np.complex128 and v.dtype == np.complex128):
        cublas_func = scikits.cuda.cublas.cublasZgemv
        alpha = np.complex128(alpha)
        beta = np.complex128(beta)
    elif (M.dtype == np.float64 and v.dtype == np.float64):
        cublas_func = scikits.cuda.cublas.cublasDgemv
        alpha = np.float64(alpha)
        beta = np.float64(beta)
    else:
        raise ValueError('unsupported combination of input types')

    incx = 1
    incy = 1
    cublas_func(handle,
                trans, m, n,
                alpha,
                M.gpudata, lda,
                v.gpudata, incx,
                beta,
                out.gpudata, incy)

def add_dot(X, Y, out, beta=0.0, handle=None): 
    if len(Y.shape) == 1:
        add_vdot(X, Y, out, beta=beta, handle=handle)
    elif len(X.shape) == 1:
        add_vdot(X, Y, out, beta=beta, transM='T', handle=handle)
    else:
        scl.add_dot(X, Y, out, beta=beta, handle=handle)

sigm_kernel = ElementwiseKernel(
    "float *x, float *z",
    "z[i] = 1.0/(1.0+exp(-x[i]))",
    "sigmoid")

def sigmoid(x, out):
    sigm_kernel(x, out)

def tanh(x, out):
    cumath.tanh(x, out)

mul_update_kernel = ElementwiseKernel(
    "float *x, float *y, float *z, float beta",
    "z[i] = beta * z[i] + x[i] * y[i]",
    "mul_update")

mul_kernel = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] * y[i]",
    "mul")

def mul(x, y, out, beta=0.0):
    if beta == 0.0:
        mul_kernel(x, y, out)
    else:
        mul_update_kernel(x, y, out, beta)

two_mul_kernel = ElementwiseKernel(
    "float c1, float *x1, float *y1, float c2, float *x2, float *y2, float *z",
    "z[i] = c1 * x1[i] * y1[i] + c2 * x2[i] * y2[i]",
    "two_mul")

def two_mul(c1, x1, y1, c2, x2, y2, out):
    two_mul_kernel(c1, x1, y1, c2, x2, y2, out)

add_kernel = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] + y[i]",
    "add")

def add(x, y, out):
    add_kernel(x, y, out)

sub_kernel = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] - y[i]",
    "sub")

def sub(x, y, out):
    sub_kernel(x, y, out)

copy_kernel = ElementwiseKernel(
    "float *x, float *y",
    "y[i] = x[i]",
    "copy")

def el_copy(x, out):
    copy_kernel(x, out)

scalar_sub_kernel = ElementwiseKernel(
    "float *x, float val, float *y",
    "y[i] = x[i] - val",
    "scalar_sub")

def scalar_sub(x, val, out):
    scalar_sub_kernel(x, val, out)

scalar_div_kernel = ElementwiseKernel(
    "float *x, float val, float *y",
    "y[i] = x[i] / val",
    "scalar_div")

def scalar_div(x, val, out):
    scalar_div_kernel(x, val, out)

def add_outer(v1, v2, M):
    if len(v1.shape) > 1 and len(v2.shape) > 1:
        add_dot(v1, v2, M, 'T', 'N', beta=beta)
        return
    if v1.shape[0] != M.shape[0] or v2.shape[0] != M.shape[1]:
        raise ValueError('size mismatch %s %s %s' %
                         (v1.shape, v2.shape, M.shape))
    m = M.shape[1]
    n = M.shape[0]

    lda = M.strides[0] // M.dtype.itemsize

    alpha = 1.0

    if (M.dtype == np.complex64 and v1.dtype == np.complex64 and v2.dtype == np.complex64):
        cublas_func = scikits.cuda.cublas.cublasCger
        alpha = np.complex64(alpha)
    elif (M.dtype == np.float32 and v1.dtype == np.float32 and v2.dtype == np.float32):
        cublas_func = scikits.cuda.cublas.cublasSger
        alpha = np.float32(alpha)
    elif (M.dtype == np.complex128 and v1.dtype == np.complex128 and v2.dtype == np.complex128):
        cublas_func = scikits.cuda.cublas.cublasZger
        alpha = np.complex128(alpha)
    elif (M.dtype == np.float64 and v1.dtype == np.float64 and v2.dtype == np.float64):
        cublas_func = scikits.cuda.cublas.cublasDger
        alpha = np.float64(alpha)
    else:
        raise ValueError('unsupported combination of input types')

    cublas_func(scm._global_cublas_handle,
                m, n,
                1.0,
                v2.gpudata, 1,
                v1.gpudata, 1,
                M.gpudata, lda)

def hstack(v1, v2, v):
    if len(v1.shape) > 1:
        if v1.shape[1] + v2.shape[1] != v.shape[1] or v1.shape[0] != v2.shape[0] or v1.shape[0] != v.shape[0]:
            raise ValueError("Dimension mismatch %s %s %s" %
                             (v1.shape, v2.shape, v.shape))
        n1 = v1.shape[1]
        n2 = v2.shape[1]
        el_copy(v1, v[..., :n1])
        el_copy(v2, v[..., n1:])
    else:
        if v1.shape[0] + v2.shape[0] != v.shape[0]:
            raise ValueError("Dimension mismatch %s %s %s" %
                             (v1.shape, v2.shape, v.shape))
        n1 = v1.shape[0]
        n2 = v2.shape[0]
        el_copy(v1, v[:n1])
        el_copy(v2, v[n1:])

def _minmax_impl(a_gpu, axis, min_or_max, out, idxout, stream=None):
    ''' Returns both max and argmax (min/argmin) along an axis.
    Hacked together from scikits.cuda code, since that doesn't have an "out"
    argument'''
    assert len(a_gpu.shape) < 3
    if axis is None:  ## Note: PyCUDA doesn't have an overall argmax/argmin!
        if min_or_max == 'max':
            return gpuarray.max(a_gpu).get()
        else:
            return gpuarray.min(a_gpu).get()
    else:
        if axis < 0:
            axis += 2
    assert axis in (0, 1)

    n, m = a_gpu.shape if a_gpu.flags.c_contiguous else (a_gpu.shape[1], a_gpu.shape[0])
    col_kernel, row_kernel = scm._get_minmax_kernel(a_gpu.dtype, min_or_max)
    target = out
    idx = idxout
    if (axis == 0 and a_gpu.flags.c_contiguous) or (axis == 1 and a_gpu.flags.f_contiguous):
        col_kernel(a_gpu, target, idx, np.uint32(m), np.uint32(n),
                   block=(32, 1, 1), grid=(m, 1, 1), stream=stream)
    else:
        row_kernel(a_gpu, target, idx, np.uint32(m), np.uint32(n),
                block=(32, 1, 1), grid=(n, 1, 1), stream=stream)

def max(a_gpu, axis, out, idxout):
    '''
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array
    axis : int (optional)
        Axis along which the maxima are computed. The default is to
        compute the maximum of the flattened array.

     Returns
    -------
    out : pycuda.gpuarray.GPUArray or float
        maxima of matrix elements along the desired axis or overall maximum.
    '''
    _minmax_impl(a_gpu, axis, "max", out, idxout)

zeroD_add_kernel = ElementwiseKernel(
    "float *x, float *y, float *z",
    "z[i] = x[i] + y[0]",
    "zeroD_add")

def zeroD_add(x, y, out):
    """
    This is terrrrribly finicky, and is not guaranteed to work with any future
    versions of pycuda. it uses the fact that elementwise kernel calls are made
    with the range of the first argument
    """
    zeroD_add_kernel(x, y, out)

def sub_matvec(M, v, axis, out):
    scm.binaryop_matvec('-',  M, v, axis, out)

def div_matvec(M, v, axis, out):
    scm.binaryop_matvec('/',  M, v, axis, out)
