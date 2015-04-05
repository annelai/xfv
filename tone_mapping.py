import math
import cv2
import numpy as np

def initGsGaussian(r):
    sigma_s = 15
#sigma_s = r/3*1.0
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

def direct_BF(E, r, gs_kernel, sigma_r):
    # E[0],[1],[2]: b,g,r
    rows, cols = E[0].shape
    Y = (E[0] + 40*E[1] + 20*E[2])/61.0
    log_Y = np.log10(Y)
    bf = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            y_begin = max(0, row-r)
            y_end = min(rows, row+r+1)
            x_begin = max(0, col-r)
            x_end = min(cols, col+r+1)
            tmp_luma = np.square( log_Y[y_begin:y_end, x_begin:x_end] - log_Y[row, col] )
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
            wacc = Gs*np.exp(-0.5*tmp_luma/sigma_r/sigma_r)
            acc_luma = np.sum(wacc*log_Y[y_begin:y_end, x_begin:x_end]) 
            wacc = np.sum(wacc)
            bf[row, col] = float(acc_luma)/float(wacc)
    return Y, log_Y, bf

def reduce_contrast(E, Y, log_Y, bf, contrast):
    detail = log_Y - bf
    min_intensity = np.amin(bf)
    max_intensity = np.amax(bf)
    print 'min', min_intensity
    print 'max', max_intensity
    delta = max_intensity - min_intensity
    gamma = math.log10(contrast) / delta
    bf_reduce = np.power(10, gamma*bf + detail)/Y
    print bf_reduce
    rows, cols = Y.shape
    output = np.zeros((rows, cols, 3))
    scale_factor = 1.0/pow(10, max_intensity*gamma)
    print 'scale_factor', scale_factor
    print 'save', 255*E[0]*bf_reduce*scale_factor
    output[:, :, 0] = (255.0*np.power(E[0]*bf_reduce*scale_factor, 1.0/2.2))
    output[:, :, 1] = (255.0*np.power(E[1]*bf_reduce*scale_factor, 1.0/2.2))
    output[:, :, 2] = (255.0*np.power(E[2]*bf_reduce*scale_factor, 1.0/2.2))
#    output[:, :, 0] = 255*E[0]*bf_reduce*scale_factor*1.2
#    output[:, :, 1] = 255*E[1]*bf_reduce*scale_factor*1.2
#    output[:, :, 2] = 255*E[2]*bf_reduce*scale_factor*1.2
    return output

def tone_map(E, radius, sigma_r, bf_method, filename):
    sigma_s = initGsGaussian(radius)
    gs_kernel = GsGaussian(radius, sigma_s)
    Y, log_Y, bf = bf_method(E, radius, gs_kernel, sigma_r)
    output = reduce_contrast(E, Y, log_Y, bf, 50)
    cv2.imwrite(filename, output)
    
    
