import numpy as np
import shutil
shuffle_vector = np.random.permutation(717)
for i in range(717):
	shutil.copy('original/images/'+str(i)+'.png', 'images/'+str(shuffle_vector[i])+'.png')
	shutil.copy('original/label/'+str(i)+'.png', 'label/'+str(shuffle_vector[i])+'.png')
