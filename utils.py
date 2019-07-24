""" This file contains different utility functions that are not connected 
in anyway to the networks presented in the tutorials, but rather help in 
processing the outputs into a more understandable way. 

For example ``tile_raster_images`` helps in generating a easy to grasp 
image from a set of samples or weights.
"""


import numpy
import pylab


def sigmoid(x):
  """Compute sigmoid 1.0/(1.0+exp(-a)) in numerically stable way."""
  x = -x
  xpos = x * (x>0.0)
  return numpy.exp(- (xpos + numpy.log(numpy.exp(x-xpos)+numpy.exp(-xpos))))


def softplus(x):
    """Compute softplus log(1+exp(x)) in numerically stable way."""
    xpos = x * (x>0.0)
    xneg = x * (x<=0.0)
    return (x>0.0) * (xpos + numpy.log(numpy.exp(-xpos) + 1)) + (x<=0.0) * numpy.log(numpy.exp(xneg)+1)


def scale_to_unit_interval(ndar,eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape,tile_spacing = (0,0), 
              scale_rows_to_unit_interval = True, output_pixel_vals = True):
    """
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can 
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.  
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
 
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image 
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype=dt)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * c
        return out_array


def pca(data, var_fraction):
    """ principal components, retaining as many components as required to 
        retain var_fraction of the variance 

    Returns projected data, projection mapping, inverse mapping, mean"""
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=1, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*var_fraction]
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    return numpy.dot(V,data), V, W


def dispims_old(M, height, width, border=0, bordercolor=0.0):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    n0 = numpy.int(pylab.ceil(numpy.sqrt(numimages)))
    n1 = numpy.int(pylab.ceil(numpy.sqrt(numimages)))
    im = bordercolor*\
         numpy.ones(((height+border)*n1+border,(width+border)*n0+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[j*(height+border)+border:(j+1)*(height+border)+border,\
                   i*(width+border)+border:(i+1)*(width+border)+border] = \
                numpy.vstack((\
                  numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),\
                         bordercolor*numpy.ones((height,border),dtype=float))),\
                  bordercolor*numpy.ones((border,width+border),dtype=float)\
                  ))
    pylab.imshow(im,cmap=pylab.cm.gray)

def gabor1d(x,mu,A,p,sigma,f,phi):
    from numpy import exp,cos
    return mu + A*exp(-(x-p)**2/(2*sigma**2)) * cos(2*3.14*f*(x-p) + phi)


def gabor2d(matrixSize, sigma, theta, omega, centerPoint):
    #sigma = scale
    #theta = orientation
    #omega = frequency    

    pointX = centerPoint[0] 
    pointY = centerPoint[1]
          
    maxX = matrixSize - pointX
    maxY = matrixSize - pointY
          
    if(numpy.mod(matrixSize, 2) ==0):
        maxX = maxX - 1
        maxY = maxY - 1

    [x, y] = numpy.meshgrid(numpy.arange(-pointX, maxX, 1), numpy.arange(-pointY, maxY, 1))
          
    e = numpy.exp(1)
            
    r = 1

    R1 = x*numpy.cos(theta) + y*numpy.sin(theta)
    R2 =-x*numpy.sin(theta) + y*numpy.cos(theta)

    expFactor = -1/2 * ( (R1/sigma)**2 + (R2/(r*sigma))**2  )

    gauss = 1 / ( numpy.sqrt(r*numpy.pi)*sigma) 
    gauss =  gauss* e**expFactor

    gaborReal = gauss* numpy.cos(omega*R1)
    gaborImag = gauss* numpy.sin(omega*R1)

    kernel = gaborReal #+ gaborImag*numpy.j
    
    return kernel

def CreateMovie(filename, plotter, numberOfFrames, fps):
    import os, sys
    import matplotlib.pyplot as plt
    for i in range(numberOfFrames):
        plotter(i)
        fname = '_tmmp%05d.png'%i
        plt.savefig(fname)
        plt.clf()
    #os.system("rm"+" filename"+".mp4")
    #os.system("ffmpeg -b 1800 -i _tmp%05d.png -r " + str(fps) + " "+filename+".mp4")
    os.system("convert -delay 10 -loop 0 _tmmp*.png "+filename+".gif")
    os.system("rm _tmmp*.png")


def dispimsmovie(filename, inv, model, patchsize, horizon, fps=5,  *args, **kwargs):
    from pylab import imshow,cm
    W = model
    W = numpy.dot(inv,W).T
    numframes = horizon
    n = W.shape[1]/numframes
    def plotter(i):
        W_ = W[:,i*n:n*(i+1)]
        image = tile_raster_images(W_, img_shape=(patchsize,patchsize), tile_shape=(12,12),tile_spacing = (1,1), scale_rows_to_unit_interval = True, output_pixel_vals = True)
        imshow(image,cmap=cm.gray,interpolation='nearest')
        pylab.axis('off')

    CreateMovie(filename, plotter, numframes, fps)


def local_normalize(data, mask): 
    from scipy import ndimage
    data_norm = numpy.zeros(data.shape,dtype=numpy.float32)       
    for i in range(data.shape[0]):
        #print ('processing image %d'% (i+1) )
        num = data[i]-ndimage.uniform_filter(data[i],mask[0])
        den = numpy.sqrt(ndimage.uniform_filter(numpy.abs(num**2)+0.0001,mask[0]))
        ln = num/den
        if numpy.isnan(ln.min()):
            assert()  
        data_norm[i]=ln
        
    return data_norm

def zca_white(X, epsilon=0.0):
    """
    Perform ZCA whitening on the matrix X.
    X is expected to be a <num_cases> by <num_dims> matrix.
    """
    m, n = X.shape

    # If there aren't more examples than dimensions then the data will
    # always lay in a lower dimensional sub-space. (3 points in 3D
    # always lie on a plance.) The smallest eigenvalues will be close
    # to zero and the covariance matrix will equal the identity
    # matrix. Check this isn't the case to avoid confusion.
    assert m > n

    # Removing the mean from each example (remove_dc) will remove one
    # degree of freedom and cause the data to lay in an n-1
    # dimensional sub-space. The smallest eigenvalue will again be
    # near zero resulting (because of numercial issues I think) in the
    # covariance matrix of the whitened data to not quite be the
    # identity matrix.

    # Compute the co-variance matrix. (Assuming each dimension already
    # has zero mean.)
    Cov = X.T.dot(X) / m
    # Compute eigenvectors (U) and eigenvalues (s).
    U, s, _ = numpy.linalg.svd(Cov)
    # Compute the whitening matrix.
    W = U.dot(numpy.diag(1. / numpy.sqrt(s + epsilon))).dot(U.T)

    return W


def CreateMovie(filename, plotter, numberOfFrames, fps):
    import os, sys
    import matplotlib.pyplot as plt
    for i in range(numberOfFrames):
        plotter(i)
        fname = '_tmp%05d.png'%i
        plt.savefig(fname)
        plt.clf()
    #os.system("ffmpeg -b 1800 -i _tmp%05d.png -r " + str(fps) + " "+filename+".avi")
    os.system("convert -delay 10 -loop 0 _tmp*.png "+filename+".gif")
    os.system("rm _tmp*.png")


def dispimsmovie(filename, W, inv = None, horizon=10, patchsize=(16,16), tile_shape=(10,10), fps=5, *args, **kwargs):
    from pylab import imshow,cm
    if inv!=None:
        W = numpy.dot(inv,W).T
    numframes = horizon
    n = W.shape[1]/numframes
    def plotter(i):
        W_ = W[:,i*n:n*(i+1)]
        image = tile_raster_images(W_, img_shape=patchsize, tile_shape=tile_shape,tile_spacing = (1,1), scale_rows_to_unit_interval = True, output_pixel_vals = True)
        imshow(image,cmap=cm.gray,interpolation='nearest')
        pylab.axis('off')

    CreateMovie(filename, plotter, numframes, fps)


def load_amat(fname, dtype=numpy.float32):
    r"""
    Load a sequence dataset from an amat-formatted file.

    This is actually a little bit dataset-specific and might not be
    completely at home here.  It is also not regularly tested.

    :notests:
    """
    rawmat = numpy.loadtxt(fname, dtype=dtype)
    mats = []
    prevcol = 0
    for col in range(rawmat.shape[0]):
        if rawmat[col,0] == -999.:
            mats.append(rawmat[prevcol:col,:])
            prevcol = col+1
    return mats    