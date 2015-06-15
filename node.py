class Node:
    """
    A Node is a compute node, which can be a CPU, a GPU (or even an FPGA in the
    long-term vision). A node does computations on Variables and stores the
    result into other Variables. Subclasses should provide concrete
    implementations of its methods. Note that all methods in a Node should be
    implemented with an optional out parameter, since pre-allocation of output
    buffers is a great way to speed up things.
    Also, Node is meant as a way to provide a uniform interface across CPU and
    GPU codes, and is *not* a threading layer.
    """
    def __init__(self, name):
        """
        Initialize this Node with a name. The name serves as an ideitifier that
        other parts of the code uses to identify this particular node, it has no
        bearing on what kind of computational substrate the Node represents
        name: the name of this node
        """
        self.name = name


    def dot_update(self, A, B, out=None, beta=0.0):
        """
        General matrix matrix product, this function only allocates memory for
        out if it is None and then delegates the meat to the concrete
        implementation. Performs
        out = beta * out + AB
        A: DataType of shape (M,N)
        B: DataType of shape (N,K)
        out: None or DataType of shape (M,K)
        beta: as defined in the formula above
        """
        outshape = A.shape[:-1] + B.shape[1:]

        if out is None:
            out = self.allocate(outshape, dtype=A.dtype)
            if beta != 0.0:
                out.fill(0.0)

        self._dot_update_impl(A, B, out, beta)
        return out

    def mul_update(self, x, y, out=None, beta=0.0):
        """
        Element-wise multiplication of two DataTypes, this function only
        allocates memory for out if it is None and then delegates the meat to
        the concrete implmentation. Performs:
        out = beta * out + x * y
        x: DataType of shape (K,)
        y: DataType of shape (K,)
        out: None or DataType of shape (K,)
        """
        if out is None:
            out = self.allocate(x.shape, dtype=x.dtype)
            if beta != 0.0:
                out.fill(0.0)

        self._mul_update_impl(x, y, out, beta)
        return out

    def add_update(self, x, y, out=None, beta=0.0):
        """
        Element-wise addition of two DataTypes, this function only allocates
        memory for out is it is None and then delegates the meat to the concrete
        implementation. Performs
        out = beta * out + x + y
        x: DataType of shape (K,)
        y: DataType of shape (K,)
        out: None or DataType of shape (K,)
        """
        if out is None:
            out = self.allocate(x.shape, dtype=x.dtype)
            if beta != 0.0:
                out.fill(0.0)

        self._add_update_impl(x, y, out, beta)
        return out


    # pure virtual methods, left for classes to implement
    def allocate(self, shape, dtype, ary=None):
        """
        Allocates and returns a DataType of given shape and dtype
        if ary is not None, then the contents of ary (a numpy.ndarray) are
        copied to the output element by element. In the case that ary is not
        None, the provided shape doesn't matter
        """
        raise NotImplementedError("Need a concrete instantiation for that")

    def transfer_data_from(self, data, other_node):
        """
        Transfer data from another node to a data object on this node
        data: data object resident on other_node
        other_node: another node, which may or may not have a shared memory with
        this node
        """
        raise NotImplementedError("Need a concrete instantiation for that")

    def _dot_update_impl(self, A, B, out, beta):
        """
        Concrete implementation of the dot product and addition, same parameters
        as dot, but out now necesarily is alocated to the correct shape
        """
        raise NotImplementedError("Need a concrete instantiation for that")

    def _mul_update_impl(self, A, B, out, beta):
        """
        Concrete implementation of the elementwise mul, same parameters as mul, but
        out now necesarily is allocated to the correct shape
        """
        raise NotImplementedError("Need a concrete instantiation for that")

    def _add_update_impl(self, A, B, out, beta):
        """
        Concrete implementation of the elementwise add, same parameters as add, but
        out now necesarily is allocated to the correct shape
        """
        raise NotImplementedError("Need a concrete instantiation for that")

