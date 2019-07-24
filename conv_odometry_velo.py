import os
import pylab
import PIL.Image
import tables
import numpy
import numpy.random
import scipy.io

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from convolutional_mlp import my_network
from utils import *
import train
import pickle

batchsize = 128
vistype = 'real'
patchsize = 16
horizon = 3
loadsize = 3000
output_folder = 'project_odometry'
n_filters = 256

print '... loading data'

datafile = tables.openFile('project_odometry/aug_conv_velo_train_data_bc01.h5','r')

data = datafile.root.images
train_features_numpy = data

# data = datafile.root.images.read()
# print data.shape
# mask = numpy.zeros((90,90),dtype=numpy.float32)
# mask[30:,:]=1.
# train_features_numpy = data*mask.reshape(1,1,1,90,90)

#labels = numpy.argmax(datafile.root.yd.read(),1).astype(numpy.int32).reshape(-1,)

labels = datafile.root.y.read()

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)


print '... instantiating model'
numpy_rng  = numpy.random.RandomState(1)
theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 15))

x = T.matrix('x').reshape((batchsize,2,5,90,90))
y = T.matrix('y')

model = my_network(numpy_rng = numpy_rng, theano_rng = theano_rng, input=x,labels=y,
                    Wl_path='/home/konda/software/python_env/bin/project_odometry/Wl4CNN256Cr.npy',
                    Wr_path='/home/konda/software/python_env/bin/project_odometry/Wr4CNN256Cr.npy',
                    image_shape=[8,90,90,batchsize], fsi=[8,16,16,n_filters],kernel_stride=(5,5),
                    single_channel=False, vistype = vistype, nout=3)


# model.load('odometry_conv_acc_unsup_st_tanh.npz')

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


def dispimsmovie(filename, model, fps=5, *args, **kwargs):
    from pylab import imshow,cm
    Wl = model.Wl.get_value(borrow=True)
    Wr = model.Wr.get_value(borrow=True)
    Wl = Wl.transpose(3,0,1,2)[:,:5].reshape(256,-1)
    Wr = Wr.transpose(3,0,1,2)[:,:5].reshape(256,-1)

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
learningrate = 0.003

trainer = train.GraddescentMinibatch_with_labels(model, train_features_numpy, labels, batchsize, loadsize, learningrate)

for epoch in xrange(numepochs):   
    trainer.step()
    if epoch%10 == 0:
        # print ('test error %f %%'%(model.get_errors_batchwise(test_data,test_labels)*100.))
        # dispimsmovie('W_filter_odometry_conv_velo', model, horizon)       
        model.save_npz('odometry_conv_acc_unsup_st_tanh')

