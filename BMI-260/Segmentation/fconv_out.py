## Function for computing forward pass on a fully convolutional network using 'shift and stitch' interpolation
## Adapted from jancowczyk by yr897021

import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import argparse
import os
import glob 
from skimage.color import rgb2gray
import caffe

def fconv_out(pattern, **kwargs):

    list_flag = kwargs.pop('list_flag', False)
    patchsize = kwargs.pop('patchsize', 64)
    displace = kwargs.pop('displace', 4)
    gray = kwargs.pop('displace', False)
    outdir = kwargs.pop('outdir', './output/')
    resize = kwargs.pop('resize', 1)
    binary = kwargs.pop('mean', "DB/DB_1_64_image_mean_clean")
    model = kwargs.pop('model', "full_convolutional_net.caffemodel")
    deploy = kwargs.pop('deploy', "deploy.prototxt")
    gpuid = kwargs.pop('gpuid', 0)
    stride = kwargs.pop('stride', 1)

    # Check if output path exists
    OUTPUT_DIR=outdir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    #load fully convolutional network 
    net_full_conv = caffe.Net(deploy, model, caffe.TEST)

    #load our mean file and get it into the right shape
    transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})

    a = caffe.io.caffe_pb2.BlobProto()
    file = open(binary,'rb')
    data = file.read()
    a.ParseFromString(data)
    means = a.data
    means = np.asarray(means)
    mean_size=means.shape[0]   

    if (gray): #allows for mean file which is of a different size than a patch
        mean_size=int(np.sqrt(mean_size))
        means = means.reshape(1, mean_size, mean_size)
    else:
        mean_size=int(np.sqrt(mean_size/3))
        means = means.reshape(3, mean_size, mean_size)

    transformer.set_mean('data',means.mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)

    if(not gray):
        transformer.set_channel_swap('data', (2,1,0))

    #set the mode to use the GPU
    caffe.set_device(gpuid)
    caffe.set_mode_gpu()

    print 'Pulling out filenames'
    if (list_flag):
        files = pattern
    else:
        files = sorted(glob.glob(pattern))

    results = []
    for fname in files:

        fname=fname.strip()
        newfname_prob = "%s/%s_prob_%s.png" % (OUTPUT_DIR,os.path.basename(fname)[0:-4],
                            model.split('/')[-1]) 

        if (os.path.exists(newfname_prob)):
            print newfname_prob + ' <------------------------ ' 
        print "working on file: \t %s" % fname

        outputimage = np.zeros(shape=(10, 10))
        scipy.misc.imsave(newfname_prob, outputimage)

        im_orig = caffe.io.load_image(fname)
        im_orig = caffe.io.resize_image(im_orig, [round(im_orig.shape[0] / resize), round(im_orig.shape[1] / resize)])

        print im_orig.shape

        nrow_in=im_orig.shape[0] #we'll be doing padding later, 
        ncol_in=im_orig.shape[1] #so lets make sure we know the original size

        patch_size = patchsize #the patch size that trained the network
        hpatch_size = patch_size / 2 #this is the size of the edges around the image

        displace_factor = displace

        im_orig = np.lib.pad(im_orig, ((hpatch_size, hpatch_size+displace_factor),(hpatch_size, hpatch_size+displace_factor),(0, 0)),  'symmetric')

        print im_orig.shape

        if(gray):
            im_orig = rgb2gray(im_orig)
            im_orig = im_orig[:,:,None]

        start=time.time()

        xx_all=np.empty([0,0])
        yy_all=np.empty([0,0])
        zinter_all=np.empty([0,0])

        for r_displace in xrange(0,displace_factor,stride): # loop over the receptor field
            for c_displace in xrange(0,displace_factor,stride):
                print "Row + Col displace:\t (%d/ %d) (%d/ %d) " %( r_displace, displace_factor,c_displace, displace_factor)

                if(gray):
                    im= im_orig[0+r_displace:-displace_factor+r_displace,0+c_displace:-displace_factor+c_displace] #displace the image
                else:
                    im= im_orig[0+r_displace:-displace_factor+r_displace,0+c_displace:-displace_factor+c_displace,:] #displace the image

                print im.shape
                out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)])) #get the output 
            # only interested in the "positive class channel"
            # the negative is simply 1- this channel
                output_sub_image=out['softmax'][0][1,:,:] 

                nrow_out=output_sub_image.shape[0]
                ncol_out=output_sub_image.shape[1]

                start_spot_row=r_displace
                start_spot_col=c_displace

                end_spot_row=nrow_in+r_displace
                end_spot_col=ncol_in+c_displace

                rinter=np.linspace(start_spot_row,end_spot_row-1,num=nrow_out)
                cinter=np.linspace(start_spot_col,end_spot_col-1,num=ncol_out)

                xx,yy=np.meshgrid(cinter,rinter)

                xx_all=np.append(xx_all,xx.flatten())
                yy_all=np.append(yy_all,yy.flatten())
                zinter_all=np.append(zinter_all,output_sub_image.flatten())

                print "Time since beginning:\t %f"% (time.time()-start)
        print "Total time:\t %f"%(time.time()-start)

        start_spot_row=0
        start_spot_col=0

        end_spot_row=nrow_in
        end_spot_col=ncol_in

        xnew = np.arange(start_spot_col, end_spot_col, 1)
        ynew = np.arange(start_spot_row, end_spot_row, 1) #maybe -1?

        xx,yy=np.meshgrid(xnew,ynew)

        # 35.871207 seconds
        #interp = scipy.interpolate.NearestNDInterpolator( (xx_all,yy_all), zinter_all) 

        # 182.112707 seconds... more sophistocated linear interpolation
        interp = scipy.interpolate.LinearNDInterpolator( (xx_all,yy_all), zinter_all) 

        result0= interp(np.ravel(xx), np.ravel(yy))
        print "Total time:\t %f"%(time.time()-start)

        result0=result0.reshape(nrow_in,ncol_in)

        # Save image to disk
        scipy.misc.toimage(result0, cmin=0.0, cmax=1.0).save(newfname_prob)
        results.append(result0)
        
    return results


    



