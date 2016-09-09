import numpy as np

class Network(object):
    def __init__(self, sizes):
        # sizes is a list of size, the length of sizes is the num of layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.random.randn generate floats follow gaussian distribution
        # only nodes in the 1,2,... layers have bias and w
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]