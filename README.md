# pythonn
Python library for machine learning optimization (designed from base up for multi-CPU/GPU parallelism)
This library uses numpy, scikits.cuda, and pycuda. Hopefully, at some point all the GPU utilities should be repository-internal rather than using the scikits.cuda versions.

The focus of the project are:
a) Support multi-CPU/GPU training by default.
b) Have almost all useful functions for neural networks pre-packaged.
c) It should be possible to build a framework like distbelief easily (with easy network functions that is).
d) An attempt to create a fully functional GPUArray (for now PyCUDA + scikits.cuda seems to be only targeted towards 2D arrays).
