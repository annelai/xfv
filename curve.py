import cv
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import glob
import sys
import numpy
import random
import math

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

        mat_a_cv = cv.fromarray(mat_a)
        mat_b_cv = cv.fromarray(mat_b)
        result_cv = [  cv.fromarray(result[0]), cv.fromarray(result[1]), cv.fromarray(result[2]) ]
        cv.Solve(mat_a_cv, mat_b_cv, result_cv[color])


        
    print result[1][0:255], result[1][127:128]
    plt.plot(result[1][0:255], range(0, 255), 'ro')
    plt.show()

    return 




#main function 
if( len(sys.argv) != 2 ) :
    print 'solveCurve <image path>'
    quit()

solveCurve(str(sys.argv[1]))

#end main
       

