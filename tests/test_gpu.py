from __future__ import print_function              
import unittest
import time
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import matplotlib.pyplot as plt

mx.random.seed(1)



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

    def test_autograd(self):
        x = nd.array([[1, 2], [3, 4]])
        x.attach_grad()
        with autograd.record():
            y = 2 * x
            z = y * x

        head = nd.array([[1, 2], [3, 4]])

        y.backward(head)

        print(x.grad)

    def test_x(self):
        a = nd.ones((2,3,4))
        print(id(a))
        a[:] = a + 1
        print(a[:,:,1])
        print(id(a))

    def test_linear_regresion(self):

        data_ctx = mx.cpu()
        model_ctx = mx.cpu()

        num_inputs = 2
        num_outputs = 1
        num_examples = 10000

        def real_fn(X):
            return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

        X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
        noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
        y = real_fn(X) + noise

        batch_size = 4
        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                              batch_size=batch_size, shuffle=True)

        w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
        b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
        params = [w, b]

        for param in params:
            param.attach_grad()

        def net(X):
            return mx.nd.dot(X, w) + b

        def square_loss(yhat, y):
            return nd.mean((yhat - y) ** 2)

        def SGD(params, lr):
            for param in params:
                param[:] = param - lr * param.grad

        epochs = 10
        learning_rate = .0001
        num_batches = num_examples/batch_size

        for e in range(epochs):
            cumulative_loss = 0
            # inner loop
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(model_ctx)
                label = label.as_in_context(model_ctx).reshape((-1, 1))
                with autograd.record():
                    output = net(data)
                    loss = square_loss(output, label)
                loss.backward()
                SGD(params, learning_rate)
                cumulative_loss += loss.asscalar()
            print(cumulative_loss / num_batches)

        print(w)
        print(b)

    def test_linear_regresion_gluon(self):
        data_ctx = mx.cpu()
        model_ctx = mx.cpu()
        num_inputs = 2
        num_outputs = 1
        num_examples = 10000

        def real_fn(X):
            return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

        X = nd.random_normal(shape=(num_examples, num_inputs))
        noise = 0.01 * nd.random_normal(shape=(num_examples,))
        y = real_fn(X) + noise

        batch_size = 4
        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                           batch_size=batch_size, shuffle=True)

        net = gluon.nn.Dense(1)

        net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

        square_loss = gluon.loss.L2Loss()

        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})

        epochs = 1
        loss_sequence = []
        num_batches = num_examples / batch_size

        for e in range(epochs):
            cumulative_loss = 0
            # inner loop
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(model_ctx)
                label = label.as_in_context(model_ctx)
                with autograd.record():
                    output = net(data)
                    loss = square_loss(output, label)
                loss.backward()
                trainer.step(batch_size)
                cumulative_loss += nd.mean(loss).asscalar()
            print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
            loss_sequence.append(cumulative_loss)

        params = net.collect_params()  # this returns a ParameterDict

        print('The type of "params" is a ', type(params))

        # A ParameterDict is a dictionary of Parameter class objects
        # therefore, here is how we can read off the parameters from it.

        for param in params.values():
            print(param.name, param.data())

    def test_mnist(self):
        data_ctx = mx.cpu()
        model_ctx = mx.cpu()

        def transform(data, label):
            return data.astype(np.float32) / 255, label.astype(np.float32)

        mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
        mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

        # image, label = mnist_train[0]
        # print(image.shape, label)

        num_inputs = 784
        num_outputs = 10
        num_examples = 60000

        batch_size = 64
        train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
        test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

        W = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
        b = nd.random_normal(shape=num_outputs, ctx=model_ctx)

        params = [W, b]

        for param in params:
            param.attach_grad()

        def softmax(y_linear):
            exp = nd.exp(y_linear)
            norms = nd.sum(exp, axis=1).reshape((-1, 1))
            return exp / norms

        # sample_y_linear = nd.random_normal(shape=(2, 10))
        # sample_yhat = softmax(sample_y_linear)
        # print(sample_yhat)
        # print(nd.sum(sample_yhat, axis=1))

        def net(X):
            y_linear = nd.dot(X, W) + b
            yhat = softmax(y_linear)
            return yhat

        def cross_entropy(yhat, y):
            return - nd.sum(y * nd.log(yhat))

        def SGD(params, lr):
            for param in params:
                param[:] = param - lr * param.grad

        def evaluate_accuracy(data_iterator, net):
            numerator = 0.
            denominator = 0.
            for i, (data, label) in enumerate(data_iterator):
                data = data.as_in_context(model_ctx).reshape((-1, 784))
                label = label.as_in_context(model_ctx)
                label_one_hot = nd.one_hot(label, 10)
                output = net(data)
                predictions = nd.argmax(output, axis=1)
                numerator += nd.sum(predictions == label)
                denominator += data.shape[0]
            return (numerator / denominator).asscalar()

        # evaluate_accuracy(test_data, net)

        epochs = 5
        learning_rate = .005

        for e in range(epochs):
            cumulative_loss = 0
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(model_ctx).reshape((-1, 784))
                label = label.as_in_context(model_ctx)
                label_one_hot = nd.one_hot(label, 10)
                with autograd.record():
                    output = net(data)
                    loss = cross_entropy(output, label_one_hot)
                loss.backward()
                SGD(params, learning_rate)
                cumulative_loss += nd.sum(loss).asscalar()

            test_accuracy = evaluate_accuracy(test_data, net)
            train_accuracy = evaluate_accuracy(train_data, net)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (
                e, cumulative_loss / num_examples, train_accuracy, test_accuracy))

        # Define the function to do prediction
        def model_predict(net,data):
            output = net(data)
            return nd.argmax(output, axis=1)

        # let's sample 10 random data points from the test set
        sample_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
        for i, (data, label) in enumerate(sample_data):
            data = data.as_in_context(model_ctx)
            print(data.shape)
            im = nd.transpose(data,(1,0,2,3))
            im = nd.reshape(im,(28,10*28,1))
            imtiles = nd.tile(im, (1,1,3))

            plt.imshow(imtiles.asnumpy())
            pred=model_predict(net,data.reshape((-1,784)))
            print('model predictions are:', pred)
            plt.show()
            break




if __name__ == '__main__':
    unittest.main()