import cv2
import numpy
import math
from curve import tone_map
import pdb


### threshold value
### result are marked by negative sign when passed threshold
def threshold(result, delta, v1):
    eps = 0.3
    if result < 0: 
        return result
    elif delta > eps:
        result = -v1
        print 'result set!'
    else:
        result = 0
    return result


### handle max itr and reverse negetive
def handle_max(result, v):
    if result == 0 :
        result = v
    elif result < 0:
        result = abs(result)
    else:
        print "Warning!!!! result invalid!!, result = ", result
    return result
        

### tone mapping(Photographic)
### second approach
def Photo_tone(img_bgr):
    print 'starting tone mapping...'
    print img_bgr.shape

    ### img_bgr is E
    

    ### convert BGR to Yxy
    X = 0.514136*img_bgr[2] + 0.323879*img_bgr[1] + 0.160364*img_bgr[0]
    Y = 0.265068*img_bgr[2] + 0.670234*img_bgr[1] + 0.064092*img_bgr[0]
    Z = 0.024119*img_bgr[2] + 0.122818*img_bgr[1] + 0.844427*img_bgr[0]
    img_xyz = numpy.array([X, Y, Z])
    W = img_xyz[0] + img_xyz[1] + img_xyz[2]
    img_yxy = numpy.array( [img_xyz[1], img_xyz[0]/W, img_xyz[1]/W] )

    #pdb.set_trace()
    print img_xyz.shape
    print img_yxy.shape
    lum_exp = img_yxy[0]
    lum = numpy.log(lum_exp+1)
    print 'lum sum ', numpy.sum(lum)
    
    ### key
    key = 0.58
    phi = 8
    ### size of gaussian blur
    g_w = 31 
    g_h = 31 
    ### threshold
    eps = 0.3
    ### max itr
    max_itr = 12

    row = len(img_bgr[0])
    col = len(img_bgr[0][0])

    ### scale to mid tone
    lum_avg = math.exp(numpy.sum(lum) / (row*col))
    lum_exp = lum_exp*key/lum_avg

    print 'midtone: avg ', lum_avg, ', lum max ', numpy.amax(lum_exp)


    ### blur size parameter
    alpha_1 = 0.35
    alpha_2 = 0.61


    ### result V
    v_result = numpy.zeros((row, col))
    V1_gaussian = numpy.zeros((max_itr, row, col))
    V2_gaussian = numpy.zeros((max_itr, row, col))
    delta_V = numpy.zeros((max_itr, row, col))
    sigma = 1.6 
    for itr in range(max_itr):

        V1_gaussian[itr] = cv2.GaussianBlur(lum_exp, (g_w, g_h), sigma*alpha_1)
        V2_gaussian[itr] = cv2.GaussianBlur(lum_exp, (g_w, g_h), sigma*alpha_2)
        delta_V[itr] = (V1_gaussian[itr]-V2_gaussian[itr])/ ((2**phi)/sigma**2 + V1_gaussian[itr])
        sigma *= 1.3
            
        print "itr ", itr, 'V1max ', numpy.amax(V1_gaussian[itr]), \
                'V2Max', numpy.amax(V2_gaussian[itr]), 'delMax', \
                numpy.amax(delta_V[itr])

    threshold_vec = numpy.vectorize(threshold)
    for itr in range(max_itr):
        v_result = threshold_vec(v_result, delta_V[itr], V1_gaussian[itr])
        print 'result_max ', numpy.amax(v_result)

    ### this is for those delta_v that are not above threshold at max itr
    handle_max_vec = numpy.vectorize(handle_max)
    v_result = handle_max_vec(v_result, V1_gaussian[max_itr-1])
    print 'result_max ', numpy.amax(v_result)


    print 'lum max ', numpy.amax(lum_exp)
        
            

    ### lower constrast
    img_yxy[0] = lum_exp/(1+v_result)
    img_yxy[0] *= 255
    #img_yxy[0] = tone_map(img_yxy[0])

    ### Yxy to BGR
    img_xyz = numpy.array([img_yxy[1]*img_yxy[0]/img_yxy[2], img_yxy[0], (1-img_yxy[1]-img_yxy[2])*(img_yxy[1]/img_yxy[2])])
    B =  0.075300*img_xyz[0] - 0.254300*img_xyz[1] + 1.189200*img_xyz[2]
    G = -1.021700*img_xyz[0] + 1.977700*img_xyz[1] + 0.043900*img_xyz[2]
    R =  2.565100*img_xyz[0] - 1.166500*img_xyz[1] - 0.398600*img_xyz[2] 
    img_bgr = numpy.array([B, G, R])
    #print img_bgr.shape

    ### clamp image
    img_bgr[img_bgr>255] = 255


    cv2.imwrite('result.jpg', cv2.merge(img_bgr))
