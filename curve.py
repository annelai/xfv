import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import glob
import sys
import numpy
import random
import math
import time

#for plotting
import matplotlib.pyplot as plt

points = 100 # points in each images
point = []   # point coordinate
smoothness = math.sqrt(2)  # curve smoothness

# random generate points
for i in range(0, points):
    row = int(random.random()*768)
    col = int(random.random()*1024)
    point.append([row,col])

# weight function 
def w(z):
    if z<0 : return 0
    if z>255 : return 255
    if z>128 : return 255-z
    return z

# for numpy adding lnE
def calE(curve, exp, *points):
    assert( len(exp) == len(points) )
    tmp = 0
    w = 0 
    for i in range(len(points)):
        tmp += w(point[i]) * (curve[point[i]] - math.log(exp[i]))
        w += w(point[i])
    return tmp/w

    


# solve non-linear curve
# images should be in path with jpg format
def solveCurve(path):
    exp_time = [] 
    cv_imgs = []
    num = 0

    #load images to cv format
    for imgName in glob.glob(path+'/*.jpg'):
        img = Image.open(imgName)
        cv_img = cv2.imread(imgName)

        # split b, g ,r 
        cv_imgs.append(cv2.split(cv_img))

        # extract exposure time
        imgInfo = img._getexif();
        for tag, value in imgInfo.items():
            if( tag == 33434 ):
                exp_time.append(value[0]/float(value[1]))
                break

        num = num + 1
    
    # empty matrix
    result = [ numpy.zeros((256+points, 1)), numpy.zeros((256+points, 1)), numpy.zeros((256+points, 1)) ]
    tmp = numpy.zeros((256+points, 1))
    
    # fill in the coefficients
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

        # g(127) = 0
        mat_a[num*points][127] = 1

        # for smoothness
        for smooth in range(1, 254):
            mat_a[num*points+smooth][smooth-1] = smoothness*w(z)
            mat_a[num*points+smooth][smooth] = -2*smoothness*w(z)
            mat_a[num*points+smooth][smooth+1] = smoothness*w(z)

        err, result[color] = cv2.solve(mat_a, mat_b, tmp,cv2.DECOMP_QR )


        
    print result[1][0:255], result[1][127:128]
    plt.plot(result[1][0:255], range(0, 255), 'ro')
    plt.show()

    return result[0:3][0:255] 

# using reconstruct curve
def radianceMap(path, curve):
    exp_time = [] 
    cv_imgs = []
    num = 0

    #load images to cv format
    for imgName in glob.glob(path+'/*.jpg'):
        img = Image.open(imgName)
        cv_img = cv2.imread(imgName)

        # split b, g ,r 
        cv_imgs.append(cv2.split(cv_img))

        # extract exposure time]
        imgInfo = img._getexif();
        for tag, value in imgInfo.items():
            if( tag == 33434 ):
                exp_time.append(value[0]/float(value[1]))
                break

        num = num + 1

    rol = len(cv_imgs[0][0])
    col = len(cv_imgs[0][0][0])
    numpy_map = [  numpy.zeros((rol, col)), numpy.zeros((rol, col)), numpy.zeros((rol, col)) ]
    # b, g ,r
#    for color in [0, 1, 2]:
#        for r in range(0, rol):
#            print r
#            for c in range(0, col):
#                ln_E = 0
#                sum_w = 1
#                #print r, c
#                # caculate weighted average
#                for pic in range(0, num):
#                    z = cv_imgs[pic][color][r][c]
##                    print z
#                    #ln_E += w(z)*(curve[color][z]-math.log(exp_time[pic]))
#                    sum_w += w(z)
#                    #print z
#                numpy_map[color][r][c] = ln_E/sum_w
    rad = []
    for color in [0, 1, 2]:
        vfunc =  numpy.vectorize(calE)
        rad[color] = vfunc( curve[color], exp_time, cv_imgs[0][color], cv_imgs[1][color], \
                                                    cv_imgs[2][color], cv_imgs[3][color], \
                                                    cv_imgs[4][color], cv_imgs[5][color], \
                                                    cv_imgs[6][color], cv_imgs[7][color], \
                                                    cv_imgs[8][color], cv_imgs[9][color], \
                                                    cv_imgs[10][color], cv_imgs[11][color], \
                                                    cv_imgs[12][color])

    # merge 3 channels back
    cv_img_result = cv2.merge(numpy_map)
    cv2.imwrite('outfile.jpg', cv_img_result)
    return numpy_map



#main function 
if( len(sys.argv) != 2 ) :
    print 'solveCurve <image path>'
    quit()

result = solveCurve(str(sys.argv[1]))
#result = numpy.zeros((256, 1))
radianceMap(str(sys.argv[1]), result)

#end main
       

