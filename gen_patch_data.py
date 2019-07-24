import os
import pylab
import PIL.Image
import tables
import numpy
import numpy.random

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import SynAE2
from utils import *
import train
from logreg import onehot

from hollywood3_tools import get_whole_clipset
from video_tools_3D import load_video_clip
from pca import whiten

batchsize = 100
vistype = 'real'
patchsize = 20
horizon = 2
loadsize = 100000
output_folder = 'project_odometry'


files = get_whole_clipset('/home/konda/software/python_env/bin/project_odometry/file_list.txt','/home/konda/software/python_env/bin/project_odometry')


print '... opening storage file'

h5file = tables.openFile(
    'project_odometry/patch_train_data_2f.h5',
    mode='w', title='input data for ConvNet')

images = h5file.createEArray(h5file.root, 'images', tables.Float32Atom(),
                    shape=(0, 45*45*4))

velocities = h5file.createEArray(h5file.root, 'velocities', tables.Float32Atom(),
                    shape=(0, ))

chzdir = h5file.createEArray(h5file.root, 'chzdir', tables.Float32Atom(),
                    shape=(0, ))

for i in [1,2,3,4]:

	print 'preprocessing video',i

	gt = numpy.loadtxt('/home/konda/software/python_env/bin/project_odometry/poses/%02d.txt'%i)

	data_l = load_video_clip(str(files[i][0]),resize=(45,155))[:,:,55:100].reshape(-1,45*45).astype(numpy.float32)
	data_r = load_video_clip(str(files[i][1]),resize=(45,155))[:,:,55:100].reshape(-1,45*45).astype(numpy.float32)


	data = numpy.concatenate((data_l,data_r),1)
	data = numpy.concatenate((data[:-1],data[1:]),1)

	tmp_zdir = numpy.zeros((gt.shape[0],),dtype=numpy.float32)


	pt = numpy.asarray([0,0,1]).reshape(-1,1)
	for i in range(gt.shape[0]):
		x = gt[i].reshape(3,4)
		pt1 = numpy.dot(x[:3,:3],pt).reshape(-1,3) + x[:,3].reshape(-1,3)
		tmp_zdir[i] = numpy.arctan2(pt1[0,2]-x[2,3],pt1[0,0]-x[0,3])

	tmp_chzdir = (tmp_zdir[1:]-tmp_zdir[:-1])

	tmp_chzdir += (tmp_chzdir<=(-1*numpy.pi))*(2*numpy.pi)
	idx = numpy.where(tmp_chzdir>(numpy.pi))
	tmp_chzdir[idx] = 2*numpy.pi - tmp_chzdir[idx]
	tmp_chzdir = tmp_chzdir*180./numpy.pi

	tmp_velo = numpy.sqrt(( (gt[1:,[3,11]] - gt[:-1,[3,11]])**2).sum(1))


	state = numpy.random.get_state()
	numpy.random.shuffle(data)
	numpy.random.set_state(state)
	numpy.random.shuffle(tmp_chzdir)
	numpy.random.set_state(state)
	numpy.random.shuffle(tmp_velo)


	print tmp_velo.shape
	print data.shape
	print tmp_chzdir.shape
	print tmp_zdir.shape

	velocities.append(tmp_velo)
	chzdir.append(tmp_chzdir)
	images.append(data)

	h5file.flush()

h5file.close()
