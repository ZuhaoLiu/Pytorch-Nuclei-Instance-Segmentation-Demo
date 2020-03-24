import os
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from optparse import OptionParser
import torch.nn as nn
from network import DIMAN
from train_functions import train, valid_test
from data_generator import evaluate_data


def get_args():
	parser = OptionParser()
	parser.add_option('-s', '--epochs', dest='epochs', default=300, type='int',
			help='number of search epochs')
	parser.add_option('-b', '--batch-size', dest='batchsize', default=6,
			type='int', help='batch size')
	parser.add_option('-l', '--learning-rate', dest='lr', default=0.0002,
			type='long', help='learning rate')
	parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
			default=True, help='use cuda')
	parser.add_option('-c', '--load', dest='load',
			default=False, help='load file model')
	(options, args) = parser.parse_args()
	return options

if __name__ == '__main__':
	args = get_args()
	net = DIMAN().cuda()
	if args.gpu:
		cudnn.benchmark = True
		net.cuda()
	if args.load:
		net.load_state_dict(torch.load(args.load))
		print('Model loaded from {}'.format(args.load))
	try:
		train(net,
			epochs=args.epochs,
			batch_size=args.batchsize,
			lr=args.lr,
			gpu=args.gpu
			)
		test_dataset = evaluate_data(args.batchsize, image_number = 10, data_type = 'test')
		valid_test(net, 
			test_dataset,
			write = True,
			fun_type = 'test')
			
	except KeyboardInterrupt:
		torch.save(net.state_dict(), 'INTERRUPTED.pth')
		print('Saved interrupt')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)

