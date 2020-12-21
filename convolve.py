#Convolution method for image

import numpy as np
import imageio
from variables import param as par

def rf(inp):

	pot = np.zeros([inp.shape[0],inp.shape[1]])
	ran = [-2,-1,0,1,2]
	ox = 2
	oy = 2

	#Receptive field kernel
	w = [[	-.5 ,-0.125 , 0.125 ,-0.125 ,-.5],
	 	[	-0.125 ,0.125 , 0.625 ,0.125 ,-0.125],
	 	[ 	0.125 ,0.625 , 	1 ,0.625 ,0.125],
	 	[	-0.125 ,0.125 , 0.625 ,0.125 ,-0.125],
	 	[	-.5 ,-0.125 , 0.125 ,-0.125 ,-.5]]

	#Convolution
	for i in range(inp.shape[0]):
		for j in range(inp.shape[1]):
			summ = 0
			for m in ran:
				for n in ran:
					if (i+m)>=0 and (i+m)<=inp.shape[0]-1 and (j+n)>=0 and (j+n)<=inp.shape[0]-1:
						summ = summ + w[ox+m][oy+n] * inp[i+m][j+n]/255
			pot[i][j] = summ
	return pot
