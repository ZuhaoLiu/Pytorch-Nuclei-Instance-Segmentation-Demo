import numpy as np
from PIL import Image


for i in range(717): # change to your own image number
	label = np.array(Image.open('label/'+str(i)+'.png')).astype(np.float32)
	label = label / 255
	num1 = np.sum(label)
	total_number = 256*256
	num2 = total_number - num1
	weights = label*(num2/total_number) + (-(label-1))*(num1/total_number)
	np.save('fw/'+str(i)+'.npy', weights)
	
	
	
	
