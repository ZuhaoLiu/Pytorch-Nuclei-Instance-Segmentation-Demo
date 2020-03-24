import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
	def __init__(self, conv_number, channel_list):
		super(DownBlock, self).__init__()
		assert conv_number == len(channel_list) - 1, 'Channel list does not have correct length'
		self.block = nn.ModuleList()
		for i in range(conv_number):
			self.block.append(nn.Sequential(nn.Conv2d(channel_list[i], channel_list[i+1], 3, padding = 1),
							nn.ReLU()))
		self.block.append(nn.BatchNorm2d(channel_list[-1]))
		self.block.append(nn.MaxPool2d(2))
	def forward(self, x):
		for i in range(len(self.block)):
			x = self.block[i](x)
		return x
		
		

class MakeDownLayers(nn.Module):
	def __init__(self):
		super(MakeDownLayers, self).__init__()
		self.blocks = nn.ModuleList()
		conv_number = [2,2,3,3,3]
		channel_list = [[3,64,64],[64,128,128],[128,256,256,256],[256,512,512,512],[512,512,512,512]]
		for i in range(5):
			self.blocks.append(DownBlock(conv_number[i], channel_list[i]))
	def forward(self, x):
		x_list = list()
		for i in range(5):
			x = self.blocks[i](x)
			x_list.append(x)
		return x_list



class MakeUpLayers(nn.Module):
	def __init__(self):
		super(MakeUpLayers, self).__init__()
		self.conv = nn.ModuleList()
		self.upsample = nn.ModuleList()
		conv_list = [512, 512, 256, 128, 64]
		for i in range(5):
			self.conv.append(nn.Sequential(nn.Conv2d(conv_list[i], 64, 3, padding = 1),
					nn.ReLU()))
			self.upsample.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
		self.FG_classifier = nn.Sequential(nn.Conv2d(320, 160, 3, padding = 1),
						nn.ReLU(),
						nn.Conv2d(160, 2, 3, padding = 1))
		self.IN_classifier = nn.Sequential(nn.Conv2d(256, 128, 3, padding = 1),
						nn.ReLU(),
						nn.Conv2d(128, 2, 3, padding = 1),
						nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
		self.MA_classifier = nn.Sequential(nn.Conv2d(192, 96, 3, padding = 1),
						nn.ReLU(),
						nn.Conv2d(96, 2, 3, padding = 1),
						nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
	def forward(self, x_list):
		x = self.conv[0](x_list[4])
		x = self.upsample[0](x)
		for i in range(4):
			in_x = self.conv[i+1](x_list[3-i])
			x = torch.cat([in_x, x], dim = 1)
			x = self.upsample[i+1](x)
			if i == 1:
				marker = F.softmax(self.MA_classifier(x), dim = -1)
			elif i == 2:
				interval = F.softmax(self.IN_classifier(x), dim = -1)
		foreground = F.softmax(self.FG_classifier(x), dim = -1)
		return foreground, interval, marker


class DIMAN(nn.Module):
	def __init__(self):
		super(DIMAN, self).__init__()
		self.down_layers = MakeDownLayers()
		self.up_layers = MakeUpLayers()

	def forward(self, x):
		x_list = self.down_layers(x)
		foreground, interval, marker = self.up_layers(x_list)
		return foreground, interval, marker





			
			
		
			
	
		
		

