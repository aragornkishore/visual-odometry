import os
import pylab
import numpy
import numpy.random
import time 

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import *
from logreg import onehot

from convolutional_mlp import my_network as MN
from convolutional_mlp import my_network2 as MN2

from hollywood3_tools import get_whole_clipset
from video_tools_3D import load_video_clip
from pca import whiten


def build_track(velocities, chdirections):

	current_position = numpy.array((2,),dtype=numpy.float32)
	current_direction = 90.

	track = numpy.zeros((velocities.shape[0],2),dtype=numpy.float32)

	track[0,0]=0.
	track[0,1]=0.
	current_position = track[0]

	for i in range(0,velocities.shape[0]):

		current_direction += chdirections[i]

		track[i,0]=current_position[0] + velocities[i]*numpy.cos(current_direction*numpy.pi/180.)
		track[i,1]=current_position[1] + velocities[i]*numpy.sin(current_direction*numpy.pi/180.)
		#print velocities[i], chdirections[i],track[i], current_direction
		current_position = track[i,:]

	return track


def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

# def smooth(seq, window_len=3):
#     n_seq = numpy.zeros(seq.shape,dtype=numpy.int32)
#     n_seq[:window_len] = seq[:window_len]
#     for i in range(window_len,seq.shape[0]):
#     	c = numpy.argmax(numpy.bincount(seq[i-window_len:i+window_len]))
#     	n_seq[i]=c

#     n_seq[-window_len:] = seq[-window_len:]

#     return n_seq


batchsize = 100
vistype = 'real'
patchsize = 20
horizon = 2
output_folder = 'project_odometry'


files = get_whole_clipset('/home/konda/software/python_env/bin/project_odometry/file_list.txt','/home/konda/software/python_env/bin/project_odometry')


print '... instantiating model'
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

x = T.matrix('x').reshape((batchsize,3,2,90,90))
y = T.ivector('y')

model_velo = MN2(numpy_rng = numpy_rng, theano_rng = theano_rng, input=x,labels=y,
                    image_shape=[8,90,90,batchsize], fsi=[8,16,16,256], kernel_stride=(5,5),
                    single_channel=False, vistype = vistype, nout=8)
model_velo.load('project_odometry/odometry_conv_acc_unsup_st_disc.npz')

x1 = T.matrix('x1').reshape((batchsize,5,2,90,90))
y1 = T.ivector('y1')

model_chdir = MN2(numpy_rng = numpy_rng, theano_rng = theano_rng, input=x1,labels=y1,
                    image_shape=[8,90,90,batchsize], fsi=[8,16,16,256], kernel_stride=(5,5),
                    single_channel=False,vistype = vistype, nout=7)
model_chdir.load('project_odometry/odometry_conv_chdir_unsup_st_disc.npz')

Int_velo = [0.,4.,8.]
Int_chdir = [-17.,0.,17.]

gt_bins_velo = [0., 1., 2., 3., 4., 5., 6., 9., 15.]
gt_bins_chdir = [-17.1, -10., -6., -2, 2., 6.,  10.,  17.1]

bins_velo = [0.1, 1.5, 2.5, 3.5, 4.5, 5.5, 7., 11.]
bins_chdir = [-14, -8, -3, 0 , 3, 8, 14]

# err_velo=0.
# err_chdir=0.

for i in [8]:

	print 'preprocessing video',i

	gt = numpy.loadtxt('/home/konda/software/python_env/bin/project_odometry/poses/%02d.txt'%i)

	data_l = load_video_clip(str(files[i][0]))[:,:,:-10].reshape(-1,1,1,90,300).astype(numpy.float32)
	data_r = load_video_clip(str(files[i][1]))[:,:,:-10].reshape(-1,1,1,90,300).astype(numpy.float32)

	#data_l = load_video_clip(str(files[i][0]))[:,:,:].reshape(-1,1,1,90,90).astype(numpy.float32)
	#data_r = load_video_clip(str(files[i][1]))[:,:,:].reshape(-1,1,1,90,90).astype(numpy.float32)

	begin_time = time.time()

	data_l -= 84.57 #data_l.mean(0).reshape(1,1,1,90,300)
	data_r -= 84.57 #data_r.mean(0).reshape(1,1,1,90,300)

	data_l /= 155.86 #data_l.std(0).reshape(1,1,1,90,300)
	data_r /= 155.86 #data_r.std(0).reshape(1,1,1,90,300)

	data = numpy.concatenate((data_l,data_r),2)
	data = numpy.concatenate((data[:-4],data[1:-3],data[2:-2],data[3:-1],data[4:]),1)

	tmp_zdir = numpy.zeros((gt.shape[0],),dtype=numpy.float32)

	pt = numpy.asarray([0,0,1]).reshape(-1,1)
	for j in range(gt.shape[0]):
		x = gt[j].reshape(3,4)
		pt1 = numpy.dot(x[:3,:3],pt).reshape(-1,3) + x[:,3].reshape(-1,3)
		tmp_zdir[j] = numpy.arctan2(pt1[0,2]-x[2,3],pt1[0,0]-x[0,3])

	tmp_chzdir = (tmp_zdir[4:]-tmp_zdir[:-4])

	tmp_chzdir += (tmp_chzdir<=(-1*numpy.pi))*(2*numpy.pi)
	idx = numpy.where(tmp_chzdir>(numpy.pi))
	tmp_chzdir[idx] = 2*numpy.pi - tmp_chzdir[idx]
	tmp_chzdir = tmp_chzdir*180./numpy.pi
	tmp_chzdir = numpy.clip(tmp_chzdir,-17,17)
	tmp_velo = numpy.sqrt(( (gt[4:,[3,11]] - gt[:-4,[3,11]])**2).sum(1))

	gt_tmp_chzdir = tmp_chzdir
	tmp_velo = (numpy.digitize(tmp_velo,gt_bins_velo).astype(numpy.int32)-1)
	tmp_chzdir = (numpy.digitize(tmp_chzdir,gt_bins_chdir).astype(numpy.int32)-1)

	tmp_velo = numpy.asarray(bins_velo)[tmp_velo]
	tmp_chzdir = (numpy.asarray(bins_chdir)[tmp_chzdir])

	data_velo = data[:,:,:,:,110:200]
	data_velo = data_velo.transpose(0,2,1,3,4)
 
	data_chd = data[:,:,:,:,110:200]
	data_chd = data_chd.transpose(0,2,1,3,4)

	velo_scores =model_velo.get_scores_batchwise(data_velo)
	chdir_scores =model_chdir.get_scores_batchwise(data_chd)

	velo_labels = numpy.argmax(velo_scores,1)
	chdir_labels = numpy.argmax(chdir_scores,1)


	# er = (tmp_velo==velo_labels).sum()/numpy.float32(tmp_velo.shape[0])
	# ec= (tmp_chzdir==chdir_labels).sum()/numpy.float32(tmp_chzdir.shape[0])
	# err_velo += er
	# err_chdir+= ec

	# print er
	# print ec


	# pylab.hist(velo_scores.flatten(),50)
	# pylab.show()
	
	# pylab.hist(chdir_scores.flatten(),50)
	# pylab.show()
    
	velocities = numpy.asarray(bins_velo)[velo_labels] #(velo_scores * bins_velo).sum(1)
	chdirections = (numpy.asarray(bins_chdir)[chdir_labels]) #(chdir_scores * bins_chdir).sum(1)

	#chdirections = (chdir_scores + Int_chdir).mean(1) 
	#velocities   = (velo_scores  + Int_velo ).mean(1)
	#chs = smooth(chdirections)[5:-5]
	#chdirections = chs
	#ves = smooth(velocities)[5:-5]
	#velocities = ves

	idx = tmp_chzdir.shape[0]
	if idx >= 1000:
		idx = 1000

	end_time = time.time()

	print 'time taken: ',end_time-begin_time
	print 'time taken per frame: ',(end_time-begin_time)/float(data_l.shape[0])

	#pylab.plot(numpy.arange(idx),chs[:idx])
	pylab.plot(numpy.arange(idx),gt_tmp_chzdir[:idx])
	pylab.plot(numpy.arange(idx),chdirections[:idx])

	pylab.savefig(('plot_chdir%02d.pdf'%i))
	pylab.close()

	pylab.plot(numpy.arange(idx),tmp_velo[:idx])
	pylab.plot(numpy.arange(idx),velocities[:idx])

	pylab.savefig(('plot_velo%02d.pdf'%i))
	pylab.close()

	track = build_track(velocities/4.,chdirections/4.)
	gt_track = build_track((tmp_velo)/4.,tmp_chzdir/4.)
	#pylab.subplot(3,1,1)
	pylab.plot(track[:,0],track[:,1],label='Predicted path')
	#pylab.subplot(3,1,2)
	pylab.plot(gt_track[:,0],gt_track[:,1],'g:',lw=2,label='Disct. Ground truth path')
	#pylab.subplot(3,1,3)
	pylab.plot(gt[:,3],gt[:,11],'r--',label='Ground truth path')
	# pylab.show()
	pylab.legend(loc=2)
	pylab.ylim(-100,500)
	pylab.savefig(('plot%02d.pdf'%i))
	pylab.close()


