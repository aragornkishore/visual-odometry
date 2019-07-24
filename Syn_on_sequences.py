import os
import pylab
import PIL.Image
import tables
import numpy
import numpy.random
import scipy.io

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import SynAE2
from utils import *
import train
import pickle

batchsize = 100
vistype = 'real'
patchsize = 16
horizon = 5
loadsize = 100000
output_folder = 'project_odometry'


print '... loading data'

datafile = tables.openFile('project_odometry/layer1_inputdata_200_stsk_5x16x16.h5','r')

params = datafile.root.preprocessing_params
train_features_numpy = datafile.root.data_white.read()

nvis = train_features_numpy.shape[1]
n_hidden = 256

v,w,m,s = params.V[:nvis],params.W[:,:nvis],params.m0,params.s0

train_features_numpy = train_features_numpy.reshape(-1,nvis*2)

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)


print '... instantiating model'
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

x     = T.matrix('x')
model = SynAE2.SynAE(numpy_rng = numpy_rng, theano_rng = theano_rng,
                    nvis=nvis, nhid=n_hidden, vistype = vistype)

#model.load('/home/konda/software/python_env/bin/project_odometry/odometry_'+str(nvis)+'_'+str(n_hidden)+'SynAE'+'.npz')
#x = model.mappingsNonoise_batchwise(train_features_numpy,10000)
#scipy.io.savemat('mappings_500000x500.mat',{'data':x})
#pylab.hist(x.flatten(),50)
#pylab.show()

print '... done'

def CreateMovie(filename, plotter, numberOfFrames, fps):
    import os, sys
    import matplotlib.pyplot as plt
    for i in range(numberOfFrames):
        plotter(i)
        fname = '_tmp%05d.png'%i
        plt.savefig(fname)
        plt.clf()
    os.system("convert -delay 10 -loop 0 _tmp*.png "+filename+".gif")
    os.system("rm _tmp*.png")


def dispimsmovie(filename, inv, model, fps=5, *args, **kwargs):
    from pylab import imshow,cm
    Wl = model.Wl.get_value(borrow=True)
    Wr = model.Wr.get_value(borrow=True)
    Wl = numpy.dot(inv,Wl).T
    Wr = numpy.dot(inv,Wr).T
    numframes = horizon
    n = Wl.shape[1]/numframes

    W = numpy.concatenate((Wl,Wr),1).reshape(-1,Wl.shape[1])

    def plotter(i):
        W_ = W[:,i*n:n*(i+1)]
        image = tile_raster_images(W_, img_shape=(patchsize,patchsize), tile_shape=(20,20),tile_spacing = (1,1), 
                        scale_rows_to_unit_interval = True, output_pixel_vals = True)
        imshow(image,cmap=cm.gray,interpolation='nearest')
        pylab.axis('off')

    CreateMovie(filename, plotter, numframes, fps)



# TRAIN MODEL
numepochs = 1001
learningrate = 0.0001

trainer = train.GraddescentMinibatch_loadbyparts(model, train_features_numpy, batchsize, learningrate, loadsize)
for epoch in xrange(numepochs):
    trainer.step()
    if epoch%10 == 0:
        dispimsmovie('W_filter_odometry_'+str(nvis)+'_'+str(n_hidden)+'_SynAE_sk', w, model, horizon)
        model.save_npz('odometry_'+str(nvis)+'_'+str(n_hidden)+'SynAE_sk')

