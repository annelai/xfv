import math
import cv2
import numpy as np

def initGsGaussian(r):
    sigma_s = r/3*1.0
#gs_kernel = {}
#for x in range(0, 2*r+1):
#for y in range(0, 2*r+1):
#gs_kernel[x-r, y-r] = math.exp(-0.5*(x*x+y*y)/sigma_s/sigma_s)
#return gs_kernel
    return sigma_s

#def initGrGaussian(sigma_r):
#    gr_kernel = np.zeros(256)
#    for i in range(0,256):
#        gr_kernel[i] = math.exp(-0.5*i/sigma_r/sigma_r)
#    return gr_kernel

def GsGaussian(r, sigma_s):
    gs_kernel = np.zeros((2*r+1,2*r+1))
    for row in range(2*r+1):
        for col in range(2*r+1):
            gs_kernel[row, col] = math.exp(-0.5*((row-r)*(row-r)+(col-r)*(col-r))/sigma_s/sigma_s)
    return gs_kernel


def direct_BF(E, orig_img, r, gs_kernel, sigma_r):
    # E[0],[1],[2]: b,g,r
    rows, cols, ch = orig_img.shape
    bf = np.zeros((rows,cols,3))
    gr_kernel = np.vectorize(math.exp)
    for row in range(rows):
        for col in range(cols):
            y_begin = max(0, row-r)
            y_end = min(rows, row+r+1)
            x_begin = max(0, col-r)
            x_end = min(cols, col+r+1)
            tmp_r = np.square( E[2][y_begin:y_end, x_begin:x_end] - E[2][row, col] )
            tmp_g = np.square( E[1][y_begin:y_end, x_begin:x_end] - E[1][row, col] )
            tmp_b = np.square( E[0][y_begin:y_end, x_begin:x_end] - E[0][row, col] )
            tmp = tmp_r + tmp_g + tmp_b
            if (row-r >= 0 and row+r <= rows-1 and col-r >= 0 and col+r <= cols-1):
                Gs = gs_kernel[:,:]
            elif(row-r < 0 and col-r < 0):
                Gs = gs_kernel[(r-row):(2*r+1), (r-col):(2*r+1)]
            elif (row-r < 0 and col+r > cols-1):
                Gs = gs_kernel[(r-row):(2*r+1), 0:(r+cols-col)] 
            elif (row-r < 0):
                Gs = gs_kernel[(r-row):(2*r+1), 0:(2*r+1)]
            elif (row+r > rows-1 and col-r < 0):
                Gs = gs_kernel[0:(r+rows-row), (r-col):(2*r+1)]
            elif (row+r > rows-1 and col+r > cols-1):
                Gs = gs_kernel[0:(r+rows-row), 0:(r+cols-col)]
            elif (row+r > rows-1):
                Gs = gs_kernel[0:(r+rows-row), 0:(2*r+1)]
            elif (col+r > cols-1):
                Gs = gs_kernel[0:(2*r+1), 0:(r+cols-col)]
            else:
                Gs = gs_kernel[0:(2*r+1), (r-col):(2*r+1)]
            wacc = Gs*gr_kernel(-0.5*tmp/sigma_r/sigma_r/3)
            acc_r = np.sum(wacc*orig_img[y_begin:y_end, x_begin:x_end, 2])
            acc_g = np.sum(wacc*orig_img[y_begin:y_end, x_begin:x_end, 1])
            acc_b = np.sum(wacc*orig_img[y_begin:y_end, x_begin:x_end, 0])
            wacc = np.sum(wacc)
            bf[row, col, 0] = float(acc_b)/float(wacc)
            bf[row, col, 1] = float(acc_g)/float(wacc)
            bf[row, col, 2] = float(acc_r)/float(wacc)
    return bf

def bilateral_filter(E, orig_img, radius, sigma_r, bf_method, filename):
    sigma_s = initGsGaussian(radius)
    gs_kernel = GsGaussian(radius, sigma_s)
    bf = bf_method(E, orig_img, radius, gs_kernel, sigma_r)
    cv2.imwrite(filename, bf)

    
    
