import sys
from MTB import *
from curve import *
from tone_mapping import *
import timeit
from photo_map import *
#from photo_map import *

#------ Main function
# argv = image path, num_frame, ref_frame, level, radius, sigma_r, contrast
if( len(sys.argv) != 9 ) :
    print 'hdr <num_frame> <ref_frame> <MTB_level> <radius> <sigma_s> <sigma_r> <contrast: suggest 50~200> <data_set>'
    quit()
start = timeit.default_timer()
num_frame = int(sys.argv[1])
ref_frame = int(sys.argv[2])
if ref_frame >= num_frame:
    print 'invalid reference frame index!'
    quit()
level = int(sys.argv[3])
radius = int(sys.argv[4])
sigma_s = float(sys.argv[5])
sigma_r = float(sys.argv[6])
contrast = float(sys.argv[7])
data_set = str(sys.argv[8])
bf_output = 'bilateral_HDR_' + str(data_set) + '_' + str(contrast) + '.jpg'

img, exp_time = align(data_set, num_frame, ref_frame, level)
result = solveCurve(img, exp_time)
E = radianceMap(img, exp_time, result)
Photo_tone(E)
#tone_map(E, radius, sigma_s, sigma_r, direct_BF, contrast, bf_output)
tone_map(E, radius, sigma_s, sigma_r, opencv_BF, contrast, bf_output)

end = timeit.default_timer()
print 'time = ', end - start
