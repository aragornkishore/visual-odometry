import numpy, pylab
import cPickle
import warnings

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet import conv


class SynAE(object):

    def __init__(self, numpy_rng, theano_rng = None, nvis=169, nhid=256, vistype = 'real'):

        self.nvis  = nvis
        self.nhid = nhid
        self.vistype = vistype
 
        if not theano_rng : 
            theano_rng = RandomStreams(rng.randint(2**30))

        self.Wl = theano.shared(numpy.asarray(numpy_rng.uniform(
                        low=-0.05,
                        high=0.05,
                        size=(nvis,nhid)), dtype=theano.config.floatX),name ='Wl')

        self.Wr = theano.shared(numpy.asarray(numpy_rng.uniform(
                        low=-0.05,
                        high=0.05,
                        size=(nvis,nhid)), dtype=theano.config.floatX),name ='Wr')
      

        self.bvisl = theano.shared(numpy.zeros((nvis,), 
                        dtype=theano.config.floatX), name='bvisl')

        self.bvisr = theano.shared(numpy.zeros((nvis,), 
                        dtype=theano.config.floatX), name='bvisr')


        self.input = T.matrix(name = 'input')
        self.theano_rng = theano_rng

        self.input_l = self.input[:,:nvis]
        self.input_r = self.input[:,nvis:]

        self.params = [self.Wl, self.Wr, self.bvisl, self.bvisr]
 
        self.features_l,self.features_r = self.get_features(self.input_l,self.input_r)

        self.hidden = (self.features_l*self.features_r) 

        self.zl,self.zr   = self.get_reconstructed_input(self.hidden > 1.0)

        if self.vistype == 'binary':
                Ll = - T.sum( self.input_l*T.log(self.zl) + (1-self.input_l)*T.log(1-self.zl), axis=1 )
                Lr = - T.sum( self.input_r*T.log(self.zr) + (1-self.input_r)*T.log(1-self.zr), axis=1 )
        elif self.vistype == 'real':
                Ll = T.sum(0.5 * ((self.input_l - self.zl)**2), axis=1)
                Lr = T.sum(0.5 * ((self.input_r - self.zr)**2), axis=1)

        self._cost = T.mean(Ll) + T.mean(Lr) 
        self._grads = T.grad(self._cost, self.params)
        
        self.get_product = theano.function([self.input], (self.hidden > 0.)*self.hidden)

    
    def get_features(self, yl, yr):
    
        out_l = T.dot(yl,self.Wl)
        out_r = T.dot(yr,self.Wr)

        return (out_l ,out_r)

    def get_reconstructed_input(self, hidden ):    


        out_l = T.dot(hidden*self.features_r,self.Wl.T)
        if self.vistype == 'binary':
            recon_l = T.nnet.sigmoid(out_l + self.bvisl)
        elif self.vistype == 'real':
            recon_l = (out_l + self.bvisl)

        out_r = T.dot(hidden*self.features_l,self.Wr.T)    
        if self.vistype == 'binary':
            recon_r = T.nnet.sigmoid(out_r + self.bvisr)
        elif self.vistype == 'real':
            recon_r = (out_r + self.bvisr)

        return recon_l,recon_r


    def mappingsNonoise_batchwise(self, input, batchsize):
        numbatches = input.shape[0] / batchsize
        mappings = numpy.zeros((input.shape[0], self.nhid), dtype=theano.config.floatX)
        for batch in range(numbatches):
            mappings[batch*batchsize:(batch+1)*batchsize, :]=self.get_product(input[batch*batchsize:(batch+1)*batchsize])
        if numpy.mod(input.shape[0], batchsize) > 0:
            mappings[numbatches*batchsize:, :]=self.get_product(input[numbatches*batchsize:])
        return mappings


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




