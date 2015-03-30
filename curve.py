import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import glob
import sys
import numpy
import random
import math
import time

### for plotting
import matplotlib.pyplot as plt

points = 100 ### points in each images
point = []   ### point coordinate
smoothness = math.sqrt(30)  ### curve smoothness

### random generate points
for i in range(0, points):
    row = int(random.random()*768)
    col = int(random.random()*1024)
    point.append([row,col])

### weight function 
def w(z):
    if z<0 : return 0
    if z>255 : return 255
    if z>128 : return 255-z
    return z

### divide function ( prevent weight sum = 0 )
def divide(rad, w_sum):
    if w_sum == 0:
        return rad
    return rad/w_sum

### simple tone mapping
def tone_map(imgs):
    result = []
    for color in [0, 1, 2]:
        img = imgs[color]
        ### from (min, max) -> (0, max-min)
        min_val = numpy.amin(img)
        img -= min_val
        ### from (0, max-min) -> (0, 255)
        max_val = numpy.amax(img)
        img /= max_val 
        img *= 255
        ### truncate to int
        img.astype(numpy.int64)
        result.append(img)

        print 'min = ', min_val, 'max = ', max_val
        print 'after: min = ', numpy.amin(img), ', max = ', numpy.amax(img)
    return numpy.array(result) 
    


### solve non-linear curve
### images should be in path with jpg format
def solveCurve( cv_imgs, exp_time):
    num = len(cv_imgs)
   
    ### empty matrix
    result = [ numpy.zeros((256+points, 1)), numpy.zeros((256+points, 1)), numpy.zeros((256+points, 1)) ]
    tmp = numpy.zeros((256+points, 1))
    
    ### fill in the coefficients
    for color in [0, 1, 2]:
        mat_a = numpy.zeros((num*points+256, 256+points))
        mat_b = numpy.zeros((num*points+256, 1))
        for pic in range(0, num):
            for idx, pt in enumerate(point):
                z = cv_imgs[pic][color][pt[0]][pt[1]]
                print pic, color, pt, z
                weight = math.sqrt(w(z))
                mat_a[pic*points+idx][z] = weight
                mat_a[pic*points+idx][256+idx] = -1*weight
                print exp_time[pic]
                mat_b[pic*points+idx][0] = weight*math.log(exp_time[pic])

        ### g(127) = 0
        mat_a[num*points][127] = 1

        ### for smoothness
        for smooth in range(1, 254):
            mat_a[num*points+smooth][smooth-1] = smoothness*w(z)
            mat_a[num*points+smooth][smooth] = -2*smoothness*w(z)
            mat_a[num*points+smooth][smooth+1] = smoothness*w(z)

        err, result[color] = cv2.solve(mat_a, mat_b, tmp,cv2.DECOMP_QR )


        
    print result[1][0:255], result[1][127:128]
    #plt.plot(result[1][0:255], range(0, 255), 'ro')
    #plt.show()

    return result[0:3][0:255] 

### using reconstruct curve
def radianceMap(cv_imgs, exp_time, curve):
    num = len(cv_imgs)

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

        divide_vec = numpy.vectorize(divide)
        rad[color] = divide_vec(rad[color] , w_sum)
        print 'max = ', numpy.amax(rad[color])


    ### merge 3 channels back
    exp_vec = numpy.vectorize(math.exp)
    hdr = exp_vec(rad)
    print 'hdr.shape = ', hdr.shape
    final_img = tone_map(rad)
    print 'img.shape = ', final_img.shape
    cv_img_result = cv2.merge(final_img)
    cv2.imwrite('outfile.jpg', cv_img_result)
    return rad, cv_img_result 


### tone mapping(Photographic)
def Photo_tone(rad):
    
    ### key
    key = 0.18
    phi = 1
    ### size of gaussian blur
    g_w = 101
    g_h = 101
    ### threshold
    eps = 100
    ### max itr
    max_itr = 10

    for color in [0, 1, 2]:
        row = len(rad[color][0])
        col = len(rad[color][0][0])
        lum_avg = numpy.sum(rad[color])/ (row*col)
        lum = rad[color]*key/lum_avg

        ### blur size parameter
        alpha_1 = 1
        alpha_2 = 0.3


        ### result V
        v_result = numpy.zeros((row, col))
        l_d = [] ### final result
        for r in range(row):
            for c in range(col):
                r_win_start = 0
                r_win_end = g_h-1
                c_win_start = 0
                c_win_start = g_w-1
                cur_win = numpy.zeros((g_h, g_w))
                ### boundary
                r_pic_start = r-(g_h-1)/2
                c_pic_start = c-(g_w-1)/2
                r_pic_end = r+(g_h-1)/2
                c_pic_end = c+(g_w-1)/2

                ### check bound
                if r < (g_h-1)/2:
                    r_win_start = (g_h-1)/2 - r
                if (row-r) < (g_h-1)/2
                    r_win_end = (row-r) + (g_h-1)/2
                if r_pic_start < 0:
                    r_pic_start = 0
                if r_pic_end > row-1:
                    r_pic_end = row-1
                if c_pic_start < 0:
                    c_pic_start = 0
                if c_pic_end > col-1:
                    c_pic_end = col-1

                cur_win[r_win_start:r_win_end, c_win_start:c_win_end] = lum[r_pic_start:r_pic_end, c_pic_start:c_pic_end]
                sigma = 1
                for itr in range(max_itr):
                    V1_gaussian = cv2.GaussianBlur(cur_win, [g_w, g_h], sigma*alpha1)
                    V2_guassian = cv2.GaussianBlur(cur_win, [g_w, g_h], sigma*alpha2)
                    V1 = numpy.sum(V1_gaussian)
                    V2 = numpy.sum(V2_gaussian)
                    tmp_V = (V1-V2)/ ((2**phi)/sigma**2 + V1)
                    v_resut[r,c] = V1
                    if abs(tmp_V) < eps:
                        break
                
        l_d[color] = lum/(1+V1)
        l_d[color] = l_d[color]*255
        cv2.imwrite('result.jpg', l_d[color])

                
                





