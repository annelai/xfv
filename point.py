import numpy
import random

out = numpy.zeros((50, 2))
for i in range(50):
     row = int(random.random()*768)
     col = int(random.random()*1024)
     out[i] = (row, col)


numpy.savetxt('points', out)
    
