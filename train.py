import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class GraddescentMinibatch(object):

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value(borrow=True).shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) 
                            for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.input:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)



class GraddescentMinibatch_withnormalization(object):

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value(borrow=True).shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) 
                            for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.input:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            self.model.normalizefilters()

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)


class GraddescentMinibatch_varlearn(object):

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value(borrow=True).shape[0] / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) 
                            for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate[_param.name] * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.input:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)

class GraddescentMinibatch_loadbyparts(object):
 
    def __init__(self, model, data, batchsize, learningrate, loadsize, momentum=0.9, rng=None, norm = False, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.loadsize      = loadsize
        self.norm          = norm
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = loadsize / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) 
                            for p in self.model.params])
        self.load_data = theano.shared(self.data[0:loadsize])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.meannl = 0.01
        self.set_learningrate(self.learningrate)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad
            #if _param.name == self.model.nparam.name and self.norm:
            #    t1 = _param + self.incs[_param] 
            #    nl = (T.std(t1,0)+0.001)   
            #    meannl = T.mean(nl)
            #    t2 = t1 - T.mean(t1,0)
            #    self.updates[_param] = t2*(meannl/nl)
            #else:
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.input:self.load_data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for load_index in xrange(self.data.shape[0]/self.loadsize):
            self.load_data.set_value(self.data[load_index*self.loadsize:(load_index+1)*self.loadsize])
            for batch_index in self.rng.permutation(self.numbatches-1):
                stepcount += 1.0
                cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
                self._trainmodel(0)
                #print 'step: %d, cost: %f' % (stepcount, cost)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)

class GraddescentMinibatch_with_labels(object):

    def __init__(self, model, data, labels, batchsize, loadsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.labels        = labels
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = loadsize / batchsize
        self.momentum      = momentum 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng
        self.loadsize = loadsize
        self.load_data = theano.shared(self.data[0:loadsize])
        self.load_labels = theano.shared(self.labels[0:loadsize])
        self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = \
          dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) 
                            for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate, self.momentum)

    def set_learningrate(self, learningrate, momentum):
        self.learningrate = learningrate
        self.momentum = momentum
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - (1-self.momentum)*self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.input:self.load_data[self.index*self.batchsize:(self.index+1)*self.batchsize],
                          self.model.labels:self.load_labels[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0

        for load_index in xrange(self.data.shape[0]/self.loadsize):
            self.load_data.set_value(self.data[load_index*self.loadsize:(load_index+1)*self.loadsize])
            self.load_labels.set_value(self.labels[load_index*self.loadsize:(load_index+1)*self.loadsize])
            for batch_index in self.rng.permutation(self.numbatches-1):
                stepcount += 1.0
                cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
                #print cost
                self._trainmodel(0)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)

