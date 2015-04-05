import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import glob
import sys
import numpy
import random
import math
import time
import pdb

### for plotting
import matplotlib.pyplot as plt

points = 50### points in each images
point = numpy.loadtxt('points')   ### point coordinate
smoothness = math.sqrt(1000)  ### curve smoothness

### random generate points
#for i in range(0, points):
#    row = int(random.random()*768)
#    col = int(random.random()*1024)
#    point.append([row,col])

### weight function 
def w(z):
    if z<0 : return 0
    if z==0 : return 0.01
    if z>255 : return 255
    if z==255: return 0.01
    if z>128 : return 255-z
    return z

### divide function ( prevent weight sum = 0 )
def divide(rad, w_sum):
    if w_sum == 0:
        return rad
    return rad/w_sum
'''
### simple tone mapping
def tone_map(img, m_val):
    ### from (min, max) -> (0, max-min)
    min_val = numpy.amin(img)
    img -= min_val
    ### from (0, max-min) -> (0, 255)
    max_val = numpy.amax(img)
    img /= max_val 
    img *= m_val 
    ### truncate to int
    img.astype(numpy.int64)

    print 'min = ', min_val, 'max = ', max_val
    print 'after: min = ', numpy.amin(img), ', max = ', numpy.amax(img)
    return numpy.array(img) 
''' 


### solve non-linear curve
### images should be in path with jpg format
def solveCurve( cv_imgs, exp_time):
    num = len(cv_imgs)
    cv_imgs = numpy.array(cv_imgs)
    cv_imgs = numpy.swapaxes(cv_imgs, 2, 3)
    cv_imgs = numpy.swapaxes(cv_imgs, 1, 2)
    print cv_imgs.shape
    print 'solving curve using ', num, ' pics..., points = ', points, ' lambda = ', smoothness
   
    ### empty matrix
    result = [ numpy.zeros((255+points, 1)), numpy.zeros((255+points, 1)), numpy.zeros((255+points, 1)) ]
    tmp = numpy.zeros((256+points, 1))
    
    ### fill in the coefficients
    for color in [0, 1, 2]:
        mat_a = numpy.zeros((num*points+255, 256+points))
        mat_b = numpy.zeros((num*points+255, 1))
        for pic in range(0, num):
            for idx, pt in enumerate(point):
                z = cv_imgs[pic][color][pt[0]][pt[1]]
                #print pic, color, pt, z
                #weight = math.sqrt(w(z))
                weight = 1
                mat_a[pic*points+idx][z] = weight
                mat_a[pic*points+idx][256+idx] = -1*weight
                #print exp_time[pic]
                mat_b[pic*points+idx][0] = weight*math.log(exp_time[pic])

        ### g(127) = 0
        mat_a[num*points][127] = 1

        ### for smoothness
        for smooth in range(1, 254):
            #mat_a[num*points+smooth][smooth-1] = smoothness*w(z)
            #mat_a[num*points+smooth][smooth] = -2*smoothness*w(z)
            #mat_a[num*points+smooth][smooth+1] = smoothness*w(z)
            mat_a[num*points+smooth][smooth-1] = smoothness
            mat_a[num*points+smooth][smooth] = -2*smoothness
            mat_a[num*points+smooth][smooth+1] = smoothness

        err, result[color] = cv2.solve(mat_a, mat_b, result[color], cv2.DECOMP_SVD )
        numpy.savetxt('mat_a', mat_a)
        numpy.savetxt('mat_b', mat_b)


        
    #print result[1][0:255], result[1][127:128]
    #plt.plot(result[0][0:255], 'ro')
    #print result[0]
    #plt.show()
    #plt.plot(result[1][0:255], 'ro')
    #plt.show()
    #plt.plot(result[2][0:255], 'ro')
    #plt.show()
    print result[0][255]

    return result[0:3][0:256] 

### using reconstruct curve
def radianceMap(cv_imgs, exp_time, curve):
    num = len(cv_imgs)

    cv_imgs = numpy.array(cv_imgs)
    cv_imgs = numpy.swapaxes(cv_imgs, 2, 3)
    cv_imgs = numpy.swapaxes(cv_imgs, 1, 2)
    print 'cv_imgs.shape = ' , ( len(cv_imgs), len(cv_imgs[0]), len(cv_imgs[0][0]), len(cv_imgs[0][0][0]))
    row = len(cv_imgs[0][0])
    col = len(cv_imgs[0][0][0])
    rad = [  numpy.zeros((row, col)), numpy.zeros((row, col)), numpy.zeros((row, col)) ]
    w_sum = numpy.zeros((row, col))

    print 'num = ', num
    for color in [0, 1, 2]:
        curve_np = numpy.asarray(curve[color])
        for pic in range(num):
            w_vec = numpy.vectorize(w)
            ### from (col, row, 1) to (col, row)
            z_refactor = numpy.reshape(curve_np[cv_imgs[pic][color]], (row, col))
            rad[color] += w_vec(cv_imgs[pic][color])*( z_refactor - math.log(exp_time[pic]) )
            w_sum += w_vec(cv_imgs[pic][color])
            print 'processing color ' , color , ', pic ' , pic

        #pdb.set_trace()
        divide_vec = numpy.vectorize(divide)
        rad[color] = divide_vec(rad[color] , w_sum)
        print 'max = ', numpy.amax(rad[color])


    ### merge 3 channels back
    hdr = numpy.exp(rad)
    print 'hdr.shape = ', hdr.shape
    print 'red max/min', numpy.amax(hdr[2]), numpy.amin(hdr[2])
    print 'green max/min', numpy.amax(hdr[1]), numpy.amin(hdr[1])
    print 'blue max/min', numpy.amax(hdr[0]), numpy.amin(hdr[0])
    final_img = tone_map(rad, 255)
    #print 'img.shape = ', final_img.shape
    #cv_img_result = cv2.merge(final_img)
    #cv2.imwrite('outfile.jpg', cv_img_result)
    return hdr

'''
### tone mapping(Photographic)
def Photo_tone2(img_bgr):
    print 'starting tone mapping...'
    print img_bgr.shape

    ### img_bgr is ln E

    ### convert BGR to Yxy
    X = 0.412453*img_bgr[2] + 0.357580*img_bgr[1] + 0.180423*img_bgr[0]
    Y = 0.212671*img_bgr[2] + 0.715160*img_bgr[1] + 0.072169*img_bgr[0]
    Z = 0.019334*img_bgr[2] + 0.119193*img_bgr[1] + 0.950227*img_bgr[0]
    img_xyz = numpy.array([X, Y, Z])
    W = img_xyz[0] + img_xyz[1] + img_xyz[2]
    img_yxy = numpy.array( [img_xyz[1], img_xyz[0]/W, img_xyz[1]/W] )

    print img_xyz.shape
    print img_yxy.shape
    lum = img_yxy[0]
    vec_exp = numpy.vectorize(math.exp)
    lum_exp = vec_exp(lum)
    
    ### key
    key = 0.18
    phi = 8
    ### size of gaussian blur
    g_w = 31 
    g_h = 31 
    ### threshold
    eps = 0.5
    ### max itr
    max_itr = 8

    row = len(img_bgr[0])
    col = len(img_bgr[0][0])

    ### scale to mid tone
    lum_avg = math.exp(numpy.sum(lum) / (row*col))
    lum_exp = lum_exp*key/lum_avg


    ### blur size parameter
    alpha_1 = 0.35
    alpha_2 = 0.51


    ### result V
    v_result = numpy.zeros((row, col))

    for r in range(row):
        for c in range(col):
            r_win_start = 0
            r_win_end = g_h
            c_win_start = 0
            c_win_end = g_w
            cur_win = numpy.zeros((g_h, g_w))
            ### boundary
            r_pic_start = r-(g_h-1)/2
            c_pic_start = c-(g_w-1)/2
            r_pic_end = r+(g_h+1)/2
            c_pic_end = c+(g_w+1)/2

            ### check bound
            if r < (g_h-1)/2:
                r_win_start = (g_h-1)/2 - r
            if (row-r-1) < (g_h-1)/2:
                r_win_end = (row-r-1) + (g_h+1)/2
            if c < (g_w-1)/2:
                c_win_start = (g_w-1)/2 - c
            if (col-c-1) < (g_w-1)/2:
                c_win_end = (col-c-1) + (g_w+1)/2
            if r_pic_start < 0:
                r_pic_start = 0
            if r_pic_end > row:
                r_pic_end = row
            if c_pic_start < 0:
                c_pic_start = 0
            if c_pic_end > col:
                c_pic_end = col


            #print r_win_start, r_win_end, c_win_start, c_win_end
            #print r_pic_start, r_pic_end, c_pic_start, c_pic_end
            cur_win[r_win_start:r_win_end, c_win_start:c_win_end] = lum_exp[r_pic_start:r_pic_end, c_pic_start:c_pic_end]
            sigma = 0.6 
            for itr in range(max_itr):
                V1_gaussian = cv2.GaussianBlur(cur_win, (g_w, g_h), sigma*alpha_1)
                V2_gaussian = cv2.GaussianBlur(cur_win, (g_w, g_h), sigma*alpha_2)
                V1 = V1_gaussian[(g_h-1)/2, (g_w-1)/2]
                V2 = V2_gaussian[(g_h-1)/2, (g_w-1)/2]
                tmp_V = (V1-V2)/ ((2**phi)/sigma**2 + V1)
                v_result[r,c] = V1
                #if itr == max_itr-1:
                    #print 'max!!'
                if abs(tmp_V) > eps:
                    break
                sigma += 1
        #print r
            
    img_yxy[0] = lum_exp/(1+v_result)
    img_yxy[0] *= 255
    ### Yxy to BGR
    img_xyz = numpy.array([img_yxy[1]*img_yxy[0]/img_yxy[2], img_yxy[0], (1-img_yxy[1]-img_yxy[2])*(img_yxy[1]/img_yxy[2])])
    B =  0.055648*img_xyz[0] - 0.204043*img_xyz[1] + 1.057311*img_xyz[2]
    G = -0.969256*img_xyz[0] + 1.875991*img_xyz[1] + 0.041556*img_xyz[2]
    R =  3.240479*img_xyz[0] - 1.53715*img_xyz[1]  - 0.498535*img_xyz[2] 
    img_bgr = numpy.array([B, G, R])
    print img_bgr.shape

    ### clamp image
    img_bgr[img_bgr>255] = 255

    ### test 
    img_bgr[0] *= 0.8
    img_bgr[2] *= 0.8

    cv2.imwrite('result.jpg', cv2.merge(img_bgr))
'''
