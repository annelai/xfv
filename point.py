import numpy
import random

out = numpy.zeros((100, 2))
for i in range(100):
     row = int(random.random()*768)
     col = int(random.random()*1024)
     out[i] = (row, col)


numpy.savetxt('points', out)
    
