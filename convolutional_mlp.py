import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.linear.conv2d_c01b import Conv2D

from lsgd_real import LogisticRegression as LGR_real
from lsgd import LogisticRegression as LGR

class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, poolsize=(2, 2),name='layer1'):

        self.input = input

        self.W = theano.shared(numpy.asarray(
                rng.uniform(low=-0.5, high=0.5, size=filter_shape),
                dtype=theano.config.floatX), name =name+'W')

        self.b = theano.shared(value=numpy.zeros((filter_shape[3],), 
                                    dtype=theano.config.floatX),name=name+'b')

        self.transformer = Conv2D(self.W)

        conv_out = self.transformer.lmul(self.input)

        pooled_out = max_pool_c01b(c01b=conv_out, pool_shape=poolsize,
                              pool_stride=poolsize)

        features = (pooled_out + self.b.dimshuffle(0, 'x', 'x', 'x'))

        self.output = T.tanh(features)

        # store parameters of this layer
        self.params = [self.W, self.b]


class hidden_layer(object):

    def __init__(self,rng, input, nvis, nout, name='layer2'):
        
        self.nvis = nvis
        self.nout = nout
        self.input = input

        self.W = theano.shared(numpy.asarray(rng.uniform(
                    low=-0.5,
                    high=0.5,
                    size=(nvis, nout)), dtype=theano.config.floatX), name=name+'W')

        self.b = theano.shared(numpy.zeros((nout,), 
                        dtype=theano.config.floatX), name=name+'b')

        self.params = [self.W, self.b]
        
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = T.tanh(lin_output)



class my_network(object):

    def __init__(self, numpy_rng, theano_rng = None, input = None, labels=None,
                        Wl_path=None, Wr_path=None,
                        image_shape=[128,5*90*90], fsi=[8,16,16,64], kernel_stride=(6,6),
                        nout=7, dropout_prb = 0.5, 
                        single_channel=False, vistype = 'binary'):

        self.image_shape = image_shape
        self.fsi  = fsi
        self.vistype = vistype
        self.nout = nout
        self.input = input
        self.labels = labels
        self.dropout_prb = dropout_prb

        if not theano_rng : 
            theano_rng = RandomStreams(rng.randint(2**30))

        self.theano_rng = theano_rng

        if Wl_path==None:

            self.Wl = theano.shared(numpy.asarray(numpy_rng.uniform(
                            low=-0.5,
                            high=0.5,
                            size=fsi), dtype=theano.config.floatX),name ='Wl_l0')

            self.Wr = theano.shared(numpy.asarray(numpy_rng.uniform(
                            low=-0.5,
                            high=0.5   ,
                            size=fsi), dtype=theano.config.floatX),name ='Wr_l0')

        else:
            self.Wl = theano.shared(numpy.load(Wl_path),name ='Wl_l0')    
            self.Wr = theano.shared(numpy.load(Wr_path),name ='Wr_l0')

        self.trans_l = Conv2D(self.Wl,kernel_stride=kernel_stride)
        self.trans_r = Conv2D(self.Wr,kernel_stride=kernel_stride)

        input_l = self.input[:,0].dimshuffle(1,2,3,0)
        input_r = self.input[:,1].dimshuffle(1,2,3,0)


        
        if fsi[0]>3:
            concat = 3
            self.input_l = T.concatenate((input_l,T.zeros_like(input_l[0:concat,:,:,:])),axis=0)
            self.input_r = T.concatenate((input_r,T.zeros_like(input_r[0:concat,:,:,:])),axis=0)
        else:
            self.input_l = input_l
            self.input_r = input_r
            
        if single_channel==False:
            self.params_l0 = [self.Wl, self.Wr]
            self.features_l = self.trans_l.lmul(self.input_l)
            self.features_r = self.trans_r.lmul(self.input_r)
            self.product = (self.features_l*self.features_r)
        else:
            self.params_l0 = [self.Wl]
            self.features_l = self.trans_l.lmul(self.input_l)
            self.features_r = self.features_l
            self.product = (self.features_l*self.features_r)

        features = T.tanh(self.product)

        fmapshape1=int(numpy.ceil((image_shape[1]-fsi[1]+1)/float(kernel_stride[0])))

        layer1 = LeNetConvPoolLayer(numpy_rng, input=self.dropout(features),           
            filter_shape=(self.fsi[3],5,5,128), poolsize=(2, 2),name='layer1')

        layer2_input = layer1.output.dimshuffle(3,0,1,2).flatten(2)
        fmapshape2 = int(numpy.ceil((fmapshape1-5+1)/2.))

        print fmapshape1,fmapshape2
        
        layer2 = LGR_real(input=self.dropout(layer2_input), labels=self.labels, n_in=128*fmapshape2**2, n_out=nout)


        self.params = self.params_l0 + layer1.params + layer2.params

        self._cost = layer2._cost
        self._grads = T.grad(self._cost, self.params)

        self.get_l1_responses = theano.function([self.input], self.features_l*self.features_r)
        # self.get_errors = theano.function([self.input,self.labels], layer2.errors())
        # self.get_predictions = theano.function([self.input], layer2.y_pred)
        self.get_scores = theano.function([self.input], layer2.p_y_given_x)


    def dropout(self, data):

        return  self.theano_rng.binomial( size = data.shape, n = 1, p = 1-self.dropout_prb, dtype=theano.config.floatX) * data


    # def get_errors_batchwise(self,data,labels):

    #     batchsize = self.image_shape[0]
    #     numbatches = data.shape[0] / batchsize
    #     error_rate=0.
    #     for batch in range(numbatches):
    #         error_rate+=self.get_errors(data[batch*batchsize:(batch+1)*batchsize],labels[batch*batchsize:(batch+1)*batchsize])

    #     return error_rate/numpy.float32(numbatches)

    # def get_pred_batchwise(self,data):

    #     batchsize = self.image_shape[0]
    #     numbatches = data.shape[0] / batchsize
    #     preds = numpy.zeros((data.shape[0],), dtype=numpy.int32)
    #     for batch in range(numbatches):
    #         preds[batch*batchsize:(batch+1)*batchsize]=self.get_predictions(data[batch*batchsize:(batch+1)*batchsize])

    #     return preds

    def get_scores_batchwise(self,data):

        batchsize = self.image_shape[0]
        numbatches = data.shape[0] / batchsize
        scores = numpy.zeros((data.shape[0],self.nout), dtype=numpy.float32)
        for batch in range(numbatches):
            scores[batch*batchsize:(batch+1)*batchsize]=self.get_scores(data[batch*batchsize:(batch+1)*batchsize])

        return scores

        
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




class my_network2(object):

    def __init__(self, numpy_rng, theano_rng = None, input = None, labels=None,
                        Wl_path=None, Wr_path=None,
                        image_shape=[128,5*90*90], fsi=[8,16,16,64], kernel_stride=(6,6),
                        nout=7, dropout_prb = 0.5, 
                        single_channel=False, vistype = 'binary'):

        self.image_shape = image_shape
        self.fsi  = fsi
        self.vistype = vistype
        self.nout = nout
        self.input = input
        self.labels = labels
        self.dropout_prb = dropout_prb

        if not theano_rng : 
            theano_rng = RandomStreams(rng.randint(2**30))

        self.theano_rng = theano_rng

        if Wl_path==None:

            self.Wl = theano.shared(numpy.asarray(numpy_rng.uniform(
                            low=-0.5,
                            high=0.5,
                            size=fsi), dtype=theano.config.floatX),name ='Wl_l0')

            self.Wr = theano.shared(numpy.asarray(numpy_rng.uniform(
                            low=-0.5,
                            high=0.5   ,
                            size=fsi), dtype=theano.config.floatX),name ='Wr_l0')

        else:
            self.Wl = theano.shared(numpy.load(Wl_path),name ='Wl_l0')    
            self.Wr = theano.shared(numpy.load(Wr_path),name ='Wr_l0')

        self.trans_l = Conv2D(self.Wl,kernel_stride=kernel_stride)
        self.trans_r = Conv2D(self.Wr,kernel_stride=kernel_stride)

        input_l = self.input[:,0].dimshuffle(1,2,3,0)
        input_r = self.input[:,1].dimshuffle(1,2,3,0)


        
        if fsi[0]>3:
            concat = 3
            self.input_l = T.concatenate((input_l,T.zeros_like(input_l[0:concat,:,:,:])),axis=0)
            self.input_r = T.concatenate((input_r,T.zeros_like(input_r[0:concat,:,:,:])),axis=0)
        else:
            self.input_l = input_l
            self.input_r = input_r
            
        if single_channel==False:
            self.params_l0 = [self.Wl, self.Wr]
            self.features_l = self.trans_l.lmul(self.input_l)
            self.features_r = self.trans_r.lmul(self.input_r)
            self.product = (self.features_l*self.features_r)
        else:
            self.params_l0 = [self.Wl]
            self.features_l = self.trans_l.lmul(self.input_l)
            self.features_r = self.features_l
            self.product = (self.features_l*self.features_r)

        features = T.tanh(self.product)

        fmapshape1=int(numpy.ceil(image_shape[1]-fsi[1]+1)/float(kernel_stride[0]))

        layer1 = LeNetConvPoolLayer(numpy_rng, input=self.dropout(features),           
            filter_shape=(self.fsi[3],5,5,128), poolsize=(2, 2),name='layer1')

        layer2_input = layer1.output.dimshuffle(3,0,1,2).flatten(2)
        fmapshape2 = int(numpy.ceil((fmapshape1-5+1)/2.))
        
        layer2 = LGR(input=self.dropout(layer2_input), labels=self.labels, n_in=128*fmapshape2**2, n_out=nout)


        self.params = self.params_l0 + layer1.params + layer2.params

        self._cost = layer2._cost
        self._grads = T.grad(self._cost, self.params)

        self.get_l1_responses = theano.function([self.input], self.features_l*self.features_r)
        # self.get_errors = theano.function([self.input,self.labels], layer2.errors())
        # self.get_predictions = theano.function([self.input], layer2.y_pred)
        self.get_scores = theano.function([self.input], layer2.p_y_given_x)


    def dropout(self, data):

        return  self.theano_rng.binomial( size = data.shape, n = 1, p = 1-self.dropout_prb, dtype=theano.config.floatX) * data


    # def get_errors_batchwise(self,data,labels):

    #     batchsize = self.image_shape[0]
    #     numbatches = data.shape[0] / batchsize
    #     error_rate=0.
    #     for batch in range(numbatches):
    #         error_rate+=self.get_errors(data[batch*batchsize:(batch+1)*batchsize],labels[batch*batchsize:(batch+1)*batchsize])

    #     return error_rate/numpy.float32(numbatches)

    # def get_pred_batchwise(self,data):

    #     batchsize = self.image_shape[0]
    #     numbatches = data.shape[0] / batchsize
    #     preds = numpy.zeros((data.shape[0],), dtype=numpy.int32)
    #     for batch in range(numbatches):
    #         preds[batch*batchsize:(batch+1)*batchsize]=self.get_predictions(data[batch*batchsize:(batch+1)*batchsize])

    #     return preds

    def get_scores_batchwise(self,data):

        batchsize = self.image_shape[0]
        numbatches = data.shape[0] / batchsize
        scores = numpy.zeros((data.shape[0],self.nout), dtype=numpy.float32)
        for batch in range(numbatches):
            scores[batch*batchsize:(batch+1)*batchsize]=self.get_scores(data[batch*batchsize:(batch+1)*batchsize])

        return scores

        
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