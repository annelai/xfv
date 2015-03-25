import cv2
import cv
import numpy as np

num_frame = 10
ref_frame = 5
threshold = []
level = 5

def diff(a,b,bias_x,bias_y,l):
    cost = 0.0
    w = 0
    rows, cols = a.shape
    for row in range(rows):
        for col in range(cols):
            if row+bias_y >= rows or row+bias_y < 0 or col+bias_x >= cols or col+bias_x < 0:
                continue
            cost += abs(float(a[row,col])-float(b[row+bias_y,col+bias_x]))
            w += 1
    return cost/(w*1.0)


def align():
    #----- Load Image
    img = []
    img_Y = []
    for idx in range(1,num_frame+1):
        if idx < 10:
            filename = 'exposures/img0' + str(idx) + '.jpg'
        else:
            filename = 'exposures/img' + str(idx) + '.jpg'
        image = cv2.imread(filename)
        img.append(image)
    #   cv2.namedWindow("I")
    #   cv2.imshow("I", image)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
        rows, cols, ch = image.shape
        Y = cv2.cvtColor(image,cv.CV_BGR2GRAY)
        img_Y.append(Y)
    #   cv2.namedWindow("Y")
    #   cv2.imshow("Y", Y)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
    
    print 'Finding median thresholding value...'
    #----- Median thresholding
    for image in img_Y:
        histo = {}
        rows, cols = image.shape
        for row in range(rows):
            for col in range(cols):
                color = image[row, col]
                if color in histo:
                    histo[color] += 1
                else:
                    histo[color] = 1
        target = rows*cols/2
        acc = 0
        for color, freq in histo.iteritems():
            if (freq+acc) >= target:
                threshold.append(color)
            else:
                acc += freq
    
    print 'color converting...'
    #----- BGR to BW 
    img_BW = []
    for idx in range(len(img_Y)):
        rows, cols = img_Y[idx].shape
        BW = cv2.threshold(img_Y[idx], threshold[idx], 255, cv2.THRESH_BINARY)[1]
        img_BW.append(BW)
    #    cv2.namedWindow("BW")
    #    cv2.imshow("BW", BW)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    
    print 'Start alignment...'
    #----- Alignment
    for idx in range(len(img_BW)):
        for l in range(level):
            rows, cols = img_BW[ref_frame].shape
            ref = cv2.resize(img_BW[ref_frame], (int(rows*pow(2,(l-level))), int(cols*pow(2,(l-level)))) )
            if idx == ref_frame:
                continue
            current = cv2.resize(img_BW[idx], (int(rows*pow(2,(l-level))), int(cols*pow(2,(l-level)))) )
    
            # top-left
            min_cost = diff(ref,current,-1,-1,l)
            direc = [-1,-1]
            # top
            tmp_cost = diff(ref,current,0,-1,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [0,-1]
            # top-right
            tmp_cost = diff(ref,current,1,-1,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [1,-1]
            # left
            tmp_cost = diff(ref,current,-1,0,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [-1,0]
            # middle
            tmp_cost = diff(ref,current,0,0,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [0,0]
            # right
            tmp_cost = diff(ref,current,1,0,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [1,0]
            # bottom-left
            tmp_cost = diff(ref,current,-1,1,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [-1,1]
            # bottom
            tmp_cost = diff(ref,current,0,1,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [0,1]
            # bottom-right
            tmp_cost = diff(ref,current,1,1,l)
            if (tmp_cost < min_cost):
                min_cost = tmp_cost
                direc = [1,1]
             
            for row in range(rows):
                for col in range(cols):
                    if row-direc[0] >= rows or row-direc[0] < 0 or col-direc[1] >= cols or col-direc[1] < 0:
                        img_BW[idx][row,col] = 255;
                    else:
                        img_BW[idx][row,col] = img_BW[idx][row-direc[0],col-direc[1]]
        
        # Write aligned image into JPG file
        if idx > 9:
            filename = 'align_img' + str(idx) + '.jpg'
        else:
            filename = 'align_img0' + str(idx) + '.jpg'
        cv.SaveImage(filename, cv.fromarray(img_BW[idx]))
    #    cv2.namedWindow("alignment")
    #    cv2.imshow("alignment", image)
    #    cv2.waitKey(0)
    #cv2.destroyAllWindows()

align()
