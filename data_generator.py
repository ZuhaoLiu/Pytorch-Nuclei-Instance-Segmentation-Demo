import numpy as np
import torch
from crop_recover import crop_image, recover_image
import math
import watershed
import instance_metric
from PIL import Image
from scipy.io import loadmat
import os
import random
import string
import cv2
def standard(x, meanval):
	x = x.astype(np.float32)
	x = x / 255. - meanval
	return x

class train_dataset():
	def __init__(self, batch_size, image_number = 717):
		self.batch_size = batch_size
		self.image_number = image_number
		self.meanval = np.load('meanval.npy') / 255.
		self.var_name = ['self.im','self.fg','self.fw','self.iv','self.iw','self.mk','self.mw']
		exec(self.var_name[0]+'='+'np.zeros(['+str(self.image_number)+', 256, 256, 3]).astype(np.float32)')
		for i in range(6):
			exec(self.var_name[i+1]+'='+'np.zeros(['+str(self.image_number)+', 256, 256]).astype(np.float32)')
		base_path = 'dataset/train_dataset/'
		for i in range(self.image_number):
			self.im[i] = standard(np.array(Image.open(base_path+'images/'+str(i)+'.png'))[:,:,0:3], self.meanval)
			self.fg[i] = np.array(Image.open(base_path+'label/'+str(i)+'.png')).astype(np.float32)/255
			self.fw[i] = np.load(base_path+'fw/'+str(i)+'.npy')
			self.iv[i] = np.array(Image.open(base_path+'interval/'+str(i)+'.png')).astype(np.float32)
			self.iw[i] = loadmat(base_path+'iw/'+str(i)+'.mat')['Interval_weight']
			self.mk[i] = np.array(Image.open(base_path+'marker/'+str(i)+'.png')).astype(np.float32)
			self.mw[i] = loadmat(base_path+'mw/'+str(i)+'.mat')['Masker_weight']
		self.im = np.transpose(self.im, (0, 3, 1, 2))
		self.fg = np.transpose(self._make_one_hot(self.fg),(0,3,1,2))
		self.iv = np.transpose(self._make_one_hot(self.iv),(0,3,1,2))
		self.mk = np.transpose(self._make_one_hot(self.mk),(0,3,1,2))
		
		self.fw = self.fw[:,np.newaxis,:,:]
		self.iw = self.iw[:,np.newaxis,:,:]
		self.mw = self.mw[:,np.newaxis,:,:]

	def __iter__(self):
		batch_number = math.ceil(self.image_number/self.batch_size)
		for batch in range(batch_number):
			batch_im = torch.from_numpy(self.im[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			batch_fg = torch.from_numpy(self.fg[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			batch_fw = torch.from_numpy(self.fw[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			batch_iv = torch.from_numpy(self.iv[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			batch_iw = torch.from_numpy(self.iw[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			batch_mk = torch.from_numpy(self.mk[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			batch_mw = torch.from_numpy(self.mw[batch*self.batch_size: (batch+1)*self.batch_size]).cuda()
			yield [batch_im, batch_fg, batch_fw, batch_iv, batch_iw, batch_mk, batch_mw]
		return
	def _make_one_hot(self, data):
		return np.eye(2)[data.astype(np.int32)].astype(np.float32)
		


class evaluate_data():
	def __init__(self, batch_size, image_number = 10, data_type = 'valid'):
		self.data_type = data_type
		self.batch_size = batch_size
		self.meanval = np.load('meanval.npy') / 255.
		self.image_number = image_number
		self.highest_dice = 0
		self.label = np.zeros([image_number, 512, 512]).astype(np.float32)
		self.interval = np.zeros([image_number, 512, 512]).astype(np.float32)
		self.marker = np.zeros([image_number, 512, 512]).astype(np.float32)
		self.highest_dice = 0
		base_path = 'dataset/'+self.data_type+'_dataset/'
		self.crop_index = [0]
		for i in range(self.image_number):
			if i == 0:
				self.image = crop_image(standard(np.array(Image.open(base_path+'images/'+str(i)+'.png'))[:,:,0:3],
							self.meanval), 256, 256, 4)
			else:
				self.image = np.vstack((self.image,
						crop_image(standard(np.array(Image.open(base_path+'images/'+str(i)+'.png'))[:,:,0:3],
						self.meanval), 256, 256, 4)))
			self.crop_index.append(self.image.shape[0])
			self.label[i] = np.array(Image.open(base_path+'label/'+str(i)+'.png')).astype(np.float32)/255
			self.interval[i] = np.array(Image.open(base_path+'interval/'+str(i)+'.png')).astype(np.float32)
			self.marker[i] = np.array(Image.open(base_path+'marker/'+str(i)+'.png')).astype(np.float32)
		self.instance_label = self._make_instance_label()
		self.image = np.transpose(self.image, (0, 3, 1, 2))
		

	def __iter__(self):
		batch_number = math.ceil(self.crop_index[-1]/self.batch_size)
		for batch in range(batch_number):
			yield torch.from_numpy(self.image[batch*self.batch_size:(batch+1)*self.batch_size]).cuda()
		return
	def evaluate(self, pre_foreground, pre_interval, pre_marker, write = False, write_file = 'Results.txt'):
		self.write = write
		self.write_file = write_file
	
		assert pre_foreground.shape[0]==pre_interval.shape[0]==pre_marker.shape[0]==self.image.shape[0],\
			'Need to support enough predictions'
		re_foreground, re_interval, re_marker = self._recover(pre_foreground, pre_interval, pre_marker)
		seg_results = self._watershed(re_foreground, re_interval, re_marker)
		self._semantic_evaluate(re_foreground, re_interval, re_marker)
		if_save = self._instance_evaluate(seg_results)
		return if_save

	def _semantic_evaluate(self, foreground, interval, marker):
		all_dice = list()
		print_name = ['foreground', 'interval', 'marker']
		all_dice.append(self._calculate_binary_dice(foreground, self.label))
		all_dice.append(self._calculate_binary_dice(interval, self.interval))
		all_dice.append(self._calculate_binary_dice(marker, self.marker))
		print('Binary results:')
		for i in range(3):
			print('Binary dice of '+str(print_name[i])+' is '+str(all_dice[i]))
		if self.write:
			f = open(self.write_file, 'a')
			f.write('Binary results:\n')
			for i in range(3):				
				f.write('Dice of '+str(print_name[i])+' is '+str(all_dice[i])+'\n')
			f.close()

	def _instance_evaluate(self, seg_results):
		results = np.zeros([self.image_number, 5]).astype(np.float32)# First col: F1 score. 
							#Second col: precision. Third col: recall.
							#Forth col: object level dice. Fifth col: object level Hausdorff.
		print_name = ['F1-score', 'Precision rate', 'Recall rate', 'Object-level dice index', 
			'Object-level hausdorff distance']		
		F1_number = 0
		Dice_number = 0
		Haus_number = 0
		for i in range(self.image_number):
			result = instance_metric.F1score(seg_results[i], self.instance_label[i])
			if result:
				results[i,0:3] = result
				F1_number += 1
			result = instance_metric.ObjectDice(seg_results[i], self.instance_label[i])
			if result:
				results[i,3] = result
				Dice_number += 1
			#results[i, 4] = instance_metric.ObjectHausdorff(seg_results[i], self.instance_label[i])
			Haus_number += 1

		results = np.sum(results, axis = 0)
		results[0:3] = results[0:3] / F1_number
		results[3] = results[3] / Dice_number
		results[4] = results[4] / Haus_number
		print('Instance results:')
		for i in range(5):
			print(print_name[i]+': '+str(results[i]))
		if self.write:
			f = open(self.write_file, 'a')
			f.write('Instance results:\n')
			for i in range(5):
				f.write(print_name[i]+': '+str(results[i])+'\n')
			f.close()
		if results[3] > self.highest_dice and self.data_type == 'valid':
			if_save = True
			self.highest_dice = results[3]
			print('xxxxx Saved highest object-level dice {:.4f} xxxxx'.format(results[3]))
		else:
			if_save = False
		return if_save
			
	def _make_instance_label(self):
		instance_label = np.zeros(self.label.shape).astype(np.int32)
		for i in range(self.image_number):
			instance_label[i], _ = watershed.Watershed_Proposed(self.label[i].astype(np.uint8), 
									self.marker[i].astype(np.uint8))
		return instance_label
		
	def _watershed(self, foreground, interval, marker):
		foreground = (foreground - interval).astype(np.uint8)
		marker = marker.astype(np.uint8)
		seg_results = np.zeros(foreground.shape).astype(np.int32)
		for i in range(self.image_number):
			seg_results[i], _ = watershed.Watershed_Proposed(foreground[i], marker[i])
		return seg_results		

	def _recover(self, foreground, interval, marker):
		re_foreground = np.zeros(self.label.shape).astype(self.label.dtype)
		re_interval = np.zeros(self.interval.shape).astype(self.interval.dtype)
		re_marker = np.zeros(self.marker.shape).astype(self.marker.dtype)
		for i in range(self.image_number):
			re_foreground[i] = recover_image(foreground[self.crop_index[i]:self.crop_index[i+1]],
							512, 512, 256, 256, 4)
			re_interval[i] = recover_image(interval[self.crop_index[i]:self.crop_index[i+1]],
							512, 512, 256, 256, 4)
			re_marker[i] = recover_image(marker[self.crop_index[i]:self.crop_index[i+1]],
							512, 512, 256, 256, 4)
		return re_foreground, re_interval, re_marker
		
	def _calculate_binary_dice(self, image, label):
		dice = np.zeros([self.image_number]).astype(np.float32)
		for i in range(self.image_number):
			inter = np.dot(image[i].flatten(), label[i].flatten())
			union = np.sum(image[i]) + np.sum(label[i])
			dice[i] = 2 * inter / union
		return np.mean(dice)

	def color_visual(self, mask):
		output = np.zeros([mask.shape[0], mask.shape[1], 3]).astype(np.uint8)
		unique_mask = np.unique(mask)
		number = unique_mask.shape[0]
		colors = np.random.randint(0,255,[number-1,3])
		for i in range(number - 1):
			output[mask == i+1] = colors[i]
		if not os.path.exists('color_visual'):
			os.mkdir('color_visual')
		name = [random.choice(string.digits + string.ascii_letters) for i in range(16)]
		name = 'color_visual/' + ''.join(name) + '.png'
		cv2.imwrite(name,output)
		
		
		
		
			
		
			
		
				
			
			
		
		
		










