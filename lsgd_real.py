import numpy, pylab
import cPickle
import warnings

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, labels, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.input=input
        self.labels = labels

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = (T.dot(self.input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]

        self._cost = self.ABS(self.labels)
        self._grads = T.grad(self._cost, self.params)


    def MSE(self, y):
        return T.mean((0.5*(self.p_y_given_x - y)**2).sum(1))

    def ABS(self, y):
        return T.mean((T.abs_(self.p_y_given_x - y)).sum(1))


    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def updateparams_fromdict(self, newparams):
        for p in self.params:
            p.set_value(newparams[p.name])

    def get_params_dict(self):
        return dict([(p.name, p.get_value()) for p in self.params])

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def save_npz(self, filename):
        numpy.savez(filename, **(self.get_params_dict()))

    def load(self, filename):
        new_params = None
        try:
            new_params = numpy.load(filename)
        except IOError, e:
            warnings.warn('''Parameter file could not be loaded with numpy.load()!
                          Is the filename correct?\n %s''' % (e, ))
        if type(new_params) == numpy.ndarray:
            print "loading npy file"
            self.updateparams(new_params)
        elif type(new_params) == numpy.lib.npyio.NpzFile:
            print "loading npz file"
            self.updateparams_fromdict(new_params)
        else:
            warnings.warn('''Parameter file loaded, but variable type not
                          recognized. Need npz or ndarray.''', Warning)
