from optparse import OptionParser
import numpy as np
from PIL import Image
from crop_recover import *

def get_args():
	parser = OptionParser()
	parser.add_option('-e', '--cropped_image_height', dest='crop_height', default=256, 
			type='int',help='cropped image height')
	parser.add_option('-i', '--cropped_image_width', dest='crop_width', default=256,
			type='int', help='cropped image width')
	parser.add_option('-l', '--overlap_coefficient', dest='oc', default=4,
			type='int', help='how many times a image is cropped')
	(options, args) = parser.parse_args()
	return options

if __name__ == '__main__':
	'''The shape of input image is H×W×D'''
	args = get_args()
	all_index = 0
	for i in range(30):
		image = np.array(Image.open('../images/'+str(i)+'.png'))
		label = np.array(Image.open('../label/'+str(i)+'.png'))
		label = label[:,:,np.newaxis]
		cropped_images = crop_image(image, args.crop_height, args.crop_width, args.oc)
		cropped_label = crop_image(label, args.crop_height, args.crop_width, args.oc)
		keep_number = 0
		keep_list = list()
		for j in range(cropped_label.shape[0]):
			if np.sum(cropped_label[j]) != 0:
				keep_number += 1
				keep_list.append(j)
		for j in range(keep_number):
			keep_image = cropped_images[keep_list[j]]
			keep_label = cropped_label[keep_list[j]]
			keep_label = keep_label[:,:,0]
			keep_image = Image.fromarray(keep_image)
			keep_label = Image.fromarray(keep_label)
			keep_image.save('images/'+str(all_index)+'.png')
			keep_label.save('label/'+str(all_index)+'.png')
			all_index += 1
		







		
			
	
	
	
