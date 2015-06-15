class Variable:
    """
    A Variable holds the data pertaining to a variable (surprise surprise!).
    A Variable is bound to a compute node (this may be a cpu, a gpu, an fpga
    etc.  depending on the backend), and computations on this Variable has to be
    done on the given compute instance or it should be transferred to a
    different node for computation.
    Note that operations aren't directly done on Variables, bt instead the node
    inside the Variable performs operations on the data inside the Variable
    """
    def __init__(self, data=None, node=None):
        """
        Initialize member variables
        self.data is a generic data storage, can be e.g., numpy.ndarray, pycuda
        gpuarray etc.
        self.node is the compute node on which this variable is stored
        Arguments:
        data: pre-populated data storage for this Variable
        node: the node which this Variable should reside on
        """
        self.data = data
        self.node = node

    def transfer_to(self, other_node):
        """
        Transfer this Variable's memory content into the node other_node.
        Depending on whether the current and the new nodes have a shared memory
        or not, this may or may not involve an actual data transfer.
        Returns: new_variable: the transferred Variable
        """

        new_data = other_node.transfer_data_from(self.data, self.node)
        new_variable = Variable(new_data, other_node)
        return new_variable

