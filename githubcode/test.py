import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import os
import sys
import argparse
import time
import models.net_6bm as net
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob
import math
import cv2

device=torch.device('cuda:0')
enhance_net = net.enhance_net().to(device)

# Change the path if you need to load other weights.
enhance_net.load_state_dict(torch.load('weights/dehaze.pth', map_location="cpu"))

def dehaze_image(image_path):
	with torch.no_grad():
		data_hazy = Image.open(image_path)

		# Align the size so that down-sampling and up-sampling process can work normally.
		h, w = data_hazy.size
		while h>=2000 or w>=2000:
			h=int(4*h/5)
			w=int(4*w/5)
		if h%4!=0:
			h = h + (4-h%4)
		if w%4!=0:
			w = w + (4-w%4)

		data_hazy = data_hazy.resize((h,w), Image.ANTIALIAS)

		data_hazy = np.asarray(data_hazy)/255.0
		data_hazy = torch.from_numpy(data_hazy).float()
		x = data_hazy.permute(2,0,1).to(device).unsqueeze(0)
		data_hazy = data_hazy.permute(2,0,1)
		data_hazy = data_hazy.to(device).unsqueeze(0)


		torch.cuda.synchronize(device)
		start_time = time.time()
		enhanced_image = enhance_net(data_hazy)
		torch.cuda.synchronize(device)
		end_time = time.time()
		used_time = end_time - start_time
		print("used time:", used_time)
		torchvision.utils.save_image(enhanced_image, "results/" + image_path.split("/")[-1])
		return used_time

def sort_key(string):
	return int(string.split("/")[-1].split(".")[0])

# directly run "python3 test.py" to execute the program.
if __name__ == '__main__':

	test_list = glob.glob("test_images/*")
	t_list = 0
	avg_used_time = 0
	for i in range(len(test_list)):
		print(test_list[i], "is processing!")
		used_time = dehaze_image(test_list[i])
		avg_used_time += used_time / len(test_list)
	print("average time consumption:", avg_used_time)
