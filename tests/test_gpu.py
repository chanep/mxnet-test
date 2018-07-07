import unittest
import time
import mxnet as mx
from mxnet import nd


class EvaluatorTestCase(unittest.TestCase):

    def test_gpu(self):
        x = nd.random_uniform(0, 1, (10000,), mx.gpu())
        start = time.time()
        c = 0.1
        for i in range(10000):
            x += c

        t = time.time() - start
        print(x)
        print("time: " + str(t))

    def test_cpu(self):
        x = nd.random_uniform(0, 1, (10000,))
        start = time.time()
        c = 0.1
        for i in range(10000):
            x += c

        t = time.time() - start
        print(x)
        print("time: " + str(t))


if __name__ == '__main__':
    unittest.main()