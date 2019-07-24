#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import random

import cv
import numpy as np

# try:
#     import theano
#     import theano.tensor as T
#     rgbvid = T.ftensor4('rgbvid')
#     _convert_to_grayscale = T.cast(T.sum(rgbvid * [[[0.3, 0.59, 0.11]]],
#                                    axis=3),
#                                    'uint8')
#     convert_to_grayscale = theano.function(inputs=[rgbvid],
#                                            outputs=_convert_to_grayscale)
# except ImportError:
#     def convert_to_grayscale(rgb_vid):
#         return np.sum(rgb_vid * [[[0.3, 0.59, 0.11]]], axis=3).astype('uint8')


def crop_frame(video, framesize):
    """Crops frames in a video s.t. width and height are multiples of framesize
    """
    return video[:, :np.floor(video.shape[1] / framesize[0]) * framesize[0],
                 :np.floor(video.shape[2] / framesize[1]) * framesize[1]]

def get_block_from_vid(video, t, y, x, framesize, horizon):
    return video[t:t+horizon, y:y+framesize, x:x+framesize].flatten()

def load_video_clip(video_file, start_frame = 0, end_frame = None, verbose = False):
    """Loads frames from a video_clip

    Args:
        video_file: path of the video file
        start_frame: first frame to be loaded
        end_frame: last frame to be loaded

    Returns:
        A (#frames)x(height)x(width)x(#channels) NumPy array containing the
        video clip
    """
    if not os.path.exists(video_file):
        raise IOError, 'File "%s" does not exist!' % video_file
    capture = cv.CaptureFromFile(video_file)
    if not end_frame:
        end_frame = int(cv.GetCaptureProperty(capture,
                                              cv.CV_CAP_PROP_FRAME_COUNT))
    else:
        end_frame = int(min(end_frame,
                            cv.GetCaptureProperty(capture,
                                                  cv.CV_CAP_PROP_FRAME_COUNT)))
    width = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)
    if verbose:
        print "end_frame: %d" % end_frame
        print "clip has %d frames" % int(
            cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
    for _ in range(start_frame): # frames start with 1 in annotation files
        cv.GrabFrame(capture)
    frames = np.zeros((end_frame - start_frame - 2, height, width), dtype=np.uint8)
    for i in range(end_frame - start_frame - 2): # end_frame = last action frame
        img = cv.QueryFrame(capture)
        if img is None:
            continue
        tmp = cv.CreateImage(cv.GetSize(img), 8, 1)
        cv.CvtColor(img, tmp, cv.CV_BGR2GRAY)
        frames[i, :] = np.asarray(cv.GetMat(tmp))
    return np.array(frames)

def sample_clips_dense(video_l, video_r, framesize, horizon,
                       temporal_subsampling, overlap=False,
                       stride=None):
    """Gets dense samples from a video with optional overlap of 50%
    Args:
        video: the video as filepath string or numpy array
        framesize: width (=height) of one sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
        stride: iterable with 2 ints (spatial/temporal stride)
    """
    if type(video_l) == str:
        if temporal_subsampling:
            # take only every 2nd frame
            video_l = load_video_clip(video_file=video_l)[::2]
            video_r = load_video_clip(video_file=video_r)[::2]
        else:
            video_l = load_video_clip(video_file=video_l)
            video_r = load_video_clip(video_file=video_r)

    elif temporal_subsampling:
        video_l = video_l[::2]
        video_r = video_r[::2]

    #print video_l

    if video_l.shape[0]<=horizon:
        video_l = np.concatenate((video_l,video_l),0)
        video_r = np.concatenate((video_r,video_r),0)
        print "conc"

    if stride:
        nblocks_t = (video_l.shape[0] - horizon) / stride[1] + 1
        nblocks_w = (video_l.shape[2] - framesize[0]) / stride[0] + 1
        nblocks_h = (video_l.shape[1] - framesize[1]) / stride[0] + 1

        samples_l = np.zeros((nblocks_t * nblocks_w * nblocks_h,
                            framesize[0]*framesize[1]*horizon), dtype=np.uint8)
        samples_r = np.zeros((nblocks_t * nblocks_w * nblocks_h,
                            framesize[0]*framesize[1]*horizon), dtype=np.uint8)

        idx = 0
        for frame_idx in range(nblocks_t):
            for col_idx in range(nblocks_w):
                for row_idx in range(nblocks_h):
                    samples_l[idx, :] = video_l[
                        frame_idx*stride[1]:frame_idx*stride[1]+horizon,
                        row_idx*stride[0]:row_idx*stride[0]+framesize[0],
                        col_idx*stride[0]:col_idx*stride[0]+framesize[1]
                    ].flatten()
                    samples_r[idx, :] = video_r[
                        frame_idx*stride[1]:frame_idx*stride[1]+horizon,
                        row_idx*stride[0]:row_idx*stride[0]+framesize[0],
                        col_idx*stride[0]:col_idx*stride[0]+framesize[1]
                    ].flatten()

                    idx += 1
        return [samples_l,samples_r]
    else:
        video_l = crop_frame(video_l, framesize)
        video_l = video_l[:np.floor(video_l.shape[0]/horizon)*horizon]
        video_l = video_l.reshape((-1, horizon, video_l.shape[1]/framesize[0],
                            framesize[0], video_l.shape[2]/framesize[1], framesize[1]))
        video_l = video_l.transpose(0, 2, 4, 1, 3, 5).reshape((-1, horizon*framesize[0]*framesize[1]))

        video_r = crop_frame(video_r, framesize)
        video_r = video_r[:np.floor(video_r.shape[0]/horizon)*horizon]
        video_r = video_r.reshape((-1, horizon, video_r.shape[1]/framesize[0],
                            framesize[0], video_r.shape[2]/framesize[1], framesize[1]))
        video_r = video_r.transpose(0, 2, 4, 1, 3, 5).reshape((-1, horizon*framesize[0]*framesize[1]))

        return video_l,video_r

def sample_clips_random(video_l,video_r, framesize, horizon, temporal_subsampling, nsamples):
    """Gets random samples from one video
    Args:
        video: video filename or numpy array
        framesize: width (=height) of the frames in the sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
        nsamples: number of samples to extract
    Returns:
        NumPy array of shape (nsamples, horizon*framesize*framesize)
    """
    if type(video_l) == str:
        if temporal_subsampling:
            # take only every 2nd frame
            video_l = load_video_clip(video_file=str(video_l))[::2]
            video_r = load_video_clip(video_file=str(video_r))[::2]
        else:
            video_l = load_video_clip(video_file=str(video_l))
            video_r = load_video_clip(video_file=str(video_r))

    elif temporal_subsampling:
        video_l = video_l[::2]
        video_r = video_r[::2]

    print video_l.shape

    if video_l.shape[0]<=horizon:
        video_l = np.concatenate((video_l,video_l),0)
        video_r = np.concatenate((video_r,video_r),0)
        print "conc"

    video_l = crop_frame(video_l, framesize)
    video_r = crop_frame(video_r, framesize)

    block_indices = [
        (np.random.randint(video_l.shape[0]-horizon),
         np.random.randint(video_l.shape[1]-framesize[0]),
         np.random.randint(video_l.shape[2]-framesize[1])) 
        for _ in range(int(nsamples))]

    sample_l = np.vstack([video_l[frame:frame + horizon,
                            row:row + framesize[0],
                            col:col + framesize[1]].reshape(1, -1) for (
                                frame, row, col) in block_indices]
                    ).astype(np.float32)

    sample_r = np.vstack([video_r[frame:frame + horizon,
                            row:row + framesize[0],
                            col:col + framesize[1]].reshape(1, -1) for (
                                frame, row, col) in block_indices]
                    ).astype(np.float32)

    return sample_l,sample_r


def sample_clips_random_from_multiple_videos(videolist, framesize, horizon, temporal_subsampling, nsamples):
    """Gets random samples from multiple videos
    Args:
        videolist: list of video filenames
        framesize: width (=height) of the frames in the sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
        nsamples: number of samples to extract
    Returns:
        NumPy array of shape (nsamples, horizon*framesize*framesize)
    """
    assert len(videolist) > 0, 'video list is empty'
    #random.shuffle(videolist)
    # if we have more videos than the number of requested samples, we can't
    # sample from all videos
    if len(videolist) > nsamples:
        videolist = videolist[:nsamples]
    nsamples_per_clip = nsamples / len(videolist)

    samples_l = np.zeros((nsamples, horizon*framesize[0]*framesize[1]), dtype=np.uint8)
    samples_r = np.zeros((nsamples, horizon*framesize[0]*framesize[1]), dtype=np.uint8)

    for i in range(len(videolist)):
        print 'sampling from video %d of %d' % (i+1, len(videolist))

        samples_l[i*nsamples_per_clip:(i+1)*nsamples_per_clip], samples_r[i*nsamples_per_clip:(i+1)*nsamples_per_clip] = \
                sample_clips_random(str(videolist[i][0]), str(videolist[i][1]), framesize, horizon,
                                    temporal_subsampling, nsamples_per_clip)
    if nsamples % len(videolist):
        offset = (nsamples / len(videolist)) * len(videolist)
        idx = np.random.randint(len(videolist))
        samples_l[offset:],samples_r[offset:] = \
                sample_clips_random(str(videolist[idx][0]), str(videolist[idx][1]), framesize, horizon,
                                    temporal_subsampling, nsamples - offset)
    return np.vstack(samples_l),np.vstack(samples_r)

def sample_clips_dense_from_multiple_videos(videolist, framesize, horizon,
                                            temporal_subsampling, stride=None,
                                            verbose=True):
    """Gets dense samples from multiple videos
    Args:
        videolist: list of video filenames
        framesize: width (=height) of the frames in the sample
        horizon: number of frames in one sample
        temporal_subsampling: whether to skip every 2nd frame
    Returns:
        NumPy array of shape (<nsamples>, <number of blocks in vid>, horizon*framesize*framesize)
    """
    samples_l = []
    samples_r = []
    for i in range(len(videolist)):
        if verbose:
            print 'sampling from video %d of %d' % (i+1, len(videolist))
        
            L, R = sample_clips_dense(videolist[i][0], videolist[i][1], framesize, horizon,
                                temporal_subsampling, stride=stride)
            samples_l.append(L)
            samples_r.append(R)

    return np.vstack(samples_l), np.vstack(samples_r)

def get_num_subblocks(superblock_framesize, superblock_horizon, framesize, horizon, stride=None):
    if stride is None:
        stride = (framesize, horizon)
    nblocks_t = int((superblock_horizon - horizon) / stride[1] + 1)
    nblocks_w = int((superblock_framesize - framesize) / stride[0] + 1)
    nblocks_h = int((superblock_framesize - framesize) / stride[0] + 1)
    return nblocks_t * nblocks_w * nblocks_h

# if __name__ == '__main__':
#     #vid = load_video_clip('/home/vincent/data/hollywood2_isa/AVIClips05/actioncliptrain00775.avi')
#     vids = np.random.randn(10, 14, 20, 20).astype(np.float32)
#     horizon = 10
#     framesize = 16
#     samples = sample_clips_dense_from_multiple_videos(vids, framesize, horizon, False, 4)
#     print samples.shape
    #nblocks_t = vid.shape[0]/horizon
    #nblocks_w = vid.shape[2]/framesize
    #nblocks_h = vid.shape[1]/framesize
    #print 'shape of vid: %s (t,y,x)' % (vid.shape, )
    #print 'nblocks_t: %d' % (nblocks_t, )
    #print 'nblocks_w: %d' % (nblocks_w, )
    #print 'nblocks_h: %d' % (nblocks_h, )
    #samples = sample_clips_dense(vid, framesize, horizon, temporal_subsampling=False, overlap=True)
    #print 'number of samples: %s' % (samples.shape[0], )
    #import new_disptools
    #new_disptools.create_video_from_patches('/home/vincent/tmp/test_densesampling', samples[:], framesize, vid.shape[1] / (framesize/2), vid.shape[2] / (framesize/2))


# vim: set ts=4 sw=4 sts=4 expandtab:
