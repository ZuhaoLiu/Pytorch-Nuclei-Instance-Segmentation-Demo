import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_generator
from network import DIMAN

class cross_loss(nn.Module):
	def __init__(self):
		super(cross_loss, self).__init__()
		self.weights = [1,1,1]
	def forward(self, pf, pi, pm, fg, fw, iv, iw, mk, mw):
		cost_pf = -torch.mean(torch.mul(torch.sum(fg*torch.log(pf+1e-8),dim = 1),fw))*self.weights[0]
		cost_pi = -torch.mean(torch.mul(torch.sum(iv*torch.log(pi+1e-8),dim = 1),iw))*self.weights[1]
		cost_pm = -torch.mean(torch.mul(torch.sum(mk*torch.log(pm+1e-8),dim = 1),mw))*self.weights[2]
		return cost_pf + cost_pi + cost_pm


def valid_test(net, dataset, write = False, pth_name = 'CP_highest.pth', fun_type = 'valid', load_name = 'CP_highest.pth'):
	if fun_type == 'test':
		net.load_state_dict(torch.load('CP_highest.pth'))
	net.eval()
	first = True
	for num_image in dataset:
		predict, interval, marker = net(num_image)
		
		if first:
			all_predict = torch.argmax(predict, dim = 1).cpu().numpy()
			all_interval = torch.argmax(interval, dim = 1).cpu().numpy()
			all_marker = torch.argmax(marker, dim = 1).cpu().numpy()
			first = False
		else:
			all_predict = np.vstack((all_predict, torch.argmax(predict, dim = 1).cpu().numpy()))
			all_interval = np.vstack((all_interval, torch.argmax(interval, dim = 1).cpu().numpy()))
			all_marker = np.vstack((all_marker, torch.argmax(marker, dim = 1).cpu().numpy()))
	if_save = dataset.evaluate(all_predict, all_interval, all_marker, write = write)
	if if_save and fun_type == 'valid':
		torch.save(net.state_dict(), pth_name)
	

def train(net, epochs = 300, batch_size = 12, lr = 0.0002, gpu = True):
	print('''
		Starting training:
		Epochs: {}
		Batch size: {}
		Learning rate: {}
		CUDA: {}
		'''.format(epochs, batch_size, lr, str(gpu)))
	optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.8, 0.999))
	criterion = cross_loss()
	highest_dice = 0
	train_dataset = data_generator.train_dataset(batch_size)
	valid_dataset = data_generator.evaluate_data(batch_size)
	for epoch in range(epochs):
		if epoch % 5 == 0:
			valid_test(net, valid_dataset)
		print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
		epoch_loss = torch.tensor([0]).float().cuda()
		total_train_batch = 0
		net.train()
		for batch_image in train_dataset:
			pre_fg, pre_in, pre_mk = net(batch_image[0])
			loss = criterion(pre_fg, pre_in, pre_mk, *batch_image[1:])
			epoch_loss += loss
			total_train_batch += 1
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('Epoch finished ! Loss: {}'.format((epoch_loss / total_train_batch).cpu().detach().numpy()[0]))










			
	
		


