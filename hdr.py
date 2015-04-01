import sys
from MTB import *
from curve import *
from tone_mapping import *

#------ Main function
# argv = image path, num_frame, ref_frame, level
if( len(sys.argv) != 6 ) :
    print 'hdr <num_frame> <ref_frame> <level> <radius> <sigma_r>'
    quit()

num_frame = int(sys.argv[1])
ref_frame = int(sys.argv[2])
if ref_frame >= num_frame:
    print 'invalid reference frame index!'
    quit()
level = int(sys.argv[3])
radius = int(sys.argv[4])
sigma_r = float(sys.argv[5])
bf_output = 'bilateral_HDR.jpg'

img, exp_time = align(num_frame, ref_frame, level)
result = solveCurve(img, exp_time)
E = radianceMap(img, exp_time, result)
tone_map(E, radius, sigma_r, direct_BF, bf_output)
