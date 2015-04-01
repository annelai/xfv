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

'''
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
            acc_r = np.sum(wacc*E[y_begin:y_end, x_begin:x_end, 2])
            acc_g = np.sum(wacc*E[y_begin:y_end, x_begin:x_end, 1])
            acc_b = np.sum(wacc*E[y_begin:y_end, x_begin:x_end, 0])
            wacc = np.sum(wacc)
            bf[row, col, 0] = float(acc_b)/float(wacc)
            bf[row, col, 1] = float(acc_g)/float(wacc)
            bf[row, col, 2] = float(acc_r)/float(wacc)
    return bf
'''
def direct_BF(E, r, gs_kernel, sigma_r):
    # E[0],[1],[2]: b,g,r
    rows, cols, ch = E[0].shape
    Y = np.zeros((rows,cols))
    Y = 0.114*E[0] + 0.587*E[1] + 0.299*E[2]
    log_Y = math.log(I)
    bf = np.zeros((rows,cols))
    gr_kernel = np.vectorize(math.exp)
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
            wacc = Gs*gr_kernel(-0.5*tmp_luma/sigma_r/sigma_r)
            acc_luma = np.sum(wacc*log_Y[y_begin:y_end, x_begin:x_end]) 
            wacc = np.sum(wacc)
            bf[row, col] = float(acc_luma)/float(wacc)
    return Y, log_Y, bf

def reduce_contrast(E, Y, log_Y, bf, contrast):
    detail = float(log_Y) - float(bf)
    min_intensity = np.amin(bf)
    max_intensity = np.amax(bf)
    delta = max_intensity - min_intensity
    gamma = math.log(contrast) / delta
    bf_reduce = math.exp(gamma*log(bf) + detail)
    bf_reduce = bf_reduce/Y
    rows, cols = Y.shape
    output = np.zeros((rows, cols, 3))
    output[:, :, 0] = E[0]*bf_reduce
    output[:, :, 1] = E[1]*bf_reduce
    output[:, :, 2] = E[2]*bf_reduce
    return output

def tone_map(E, radius, sigma_r, bf_method, filename):
    sigma_s = initGsGaussian(radius)
    gs_kernel = GsGaussian(radius, sigma_s)
    Y, log_Y, bf = bf_method(E, radius, gs_kernel, sigma_r)
    output = reduce_contrast(E, Y, log_Y, bf, 5)
    cv2.imwrite(filename, output)
    
    
