#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np

def cliplist_from_clipset(clipset_filename, avi_dir):
    clipset_file = open(clipset_filename, 'r')
    clip_list = clipset_file.read().splitlines()
    clipset_file.close()
    clip_paths = []
    for line in clip_list:
        line = line.split(' ')
        if line[-1] == '1':
            line_tmp = line[0].split(',')
            clip_paths.append('/%s.avi,/%s.avi' % (line_tmp[0],line_tmp[1]))
    return clip_paths

def get_whole_clipset(clipset_filename, avi_dir):
    clipset_file = open(clipset_filename, 'r')
    clip_list = clipset_file.read().splitlines()
    clipset_file.close()
    clip_paths = []
    for line in clip_list:
        line = line.split(',')
        #print line[0],line[1]
        clip_paths.append(np.array(['%s/%s.avi' % (avi_dir, line[0]),'%s/%s.avi' % (avi_dir, line[1])]))

    return clip_paths

# vim: set ts=4 sw=4 sts=4 expandtab:



