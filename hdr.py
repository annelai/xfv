import cv
import cv2
import sys
from MTB import *
from curve import *

#------ Main function
# argv = image path, num_frame, ref_frame, level
if( len(sys.argv) != 5 ) :
    print 'hdr <image_path> <num_frame> <ref_frame> <level>'
    quit()

image_path = str(sys.argv[1])    
num_frame = int(sys.argv[2])
ref_frame = int(sys.argv[3])
level = int(level)
if ref_frame >= num_frame:
    print 'invalid reference frame index!'
    quit()
level = int(sys.argv[4])
result = solveCurve(str(image_path))
align(num_frame, ref_frame, level)
radianceMap(image_path, result)
