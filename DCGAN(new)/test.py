import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mping
import csv
import random
import sys


#random.seed(666)
#np.random.seed(666)
#torch.manual_seed(666)

datasize = 1

learning_rate = 0.0002
epoch_num = 1000
batch_size = 60
D_step = 10
G_step = 10


def getTags():
	text = open('tags.txt','r',encoding='utf-8')
	row = list(csv.reader(text, delimiter=','))
	haircolorset=[]
	hairlengthset=[]
	eyeset=[]
	for tag in row[0]:
		haircolorset.append(tag)
	for tag in row[1]:
		hairlengthset.append(tag)
	for tag in row[2]:
		eyeset.append(tag)
	return [haircolorset,hairlengthset,eyeset]




def getfaces():
	faces = []
	facetags = []
	text = open('filtered.txt','r',encoding='utf-8')
	row = list(csv.reader(text, delimiter=','))
	sample = np.random.permutation(18000)
	sample = sample[0:datasize]
	for i in sample:
		img=mping.imread('../../../datasets/lee_faces/selected64/'+str(i)+'.jpg')
		reform=[img[:,:,0],img[:,:,1],img[:,:,2]]
		faces.append(reform)
		hcvec = np.zeros(len(tagset[0]))
		hlvec = np.zeros(len(tagset[1]))
		eyevec = np.zeros(len(tagset[2]))
		for tag in row[i]:
			if tag in tagset[0]:
				hcvec[tagset[0].index(tag)]=1
			if tag in tagset[1]:
				hlvec[tagset[1].index(tag)]=1
			if tag in tagset[2]:
				eyevec[tagset[2].index(tag)]=1
		facetags.append(np.concatenate((hcvec,hlvec,eyevec)))
	return np.array(faces).astype(float)/255,np.array(facetags)


#when generate tags, make sure there is only one tag for each types
def GSampler(size):
	xs = torch.randn(size,80)
	tags = torch.zeros(size,26)
	for tag in tags:
		hc = random.randint(0,len(tagset[0])+5)
		if hc<len(tagset[0]):
			tag[hc]=1
		hl = random.randint(0,len(tagset[1])*2)
		if hl<len(tagset[1]):
			tag[hl+len(tagset[0])]=1
		ec = random.randint(0,len(tagset[2])+5)
		if ec<len(tagset[2]):
			tag[ec+len(tagset[0])+len(tagset[1])]=1
	return xs,tags

def DTagSampler(size):
	tags = torch.zeros(size,26)
	for tag in tags:
		hc = random.randint(0,len(tagset[0])+5)
		if hc<len(tagset[0]):
			tag[hc]=1
		hl = random.randint(0,len(tagset[1])*2)
		if hl<len(tagset[1]):
			tag[hl+len(tagset[0])]=1
		ec = random.randint(0,len(tagset[2])+5)
		if ec<len(tagset[2]):
			tag[ec+len(tagset[0])+len(tagset[1])]=1
	return tags


#for generator, input is a norm-distribution vector x[0...100] and random tags[0...26]
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.fc = nn.Linear(80,4*4*512)
		self.remap = nn.Sequential(
			nn.ConvTranspose2d(512,256,4,stride=2,padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.ConvTranspose2d(64,3,4,stride=2,padding=1),
			)

	def forward(self, x):
		c = self.fc(x)
		c = c.view(c.shape[0],512,4,4)
		c = self.remap(c)
		c = nn.Tanh()(c)
		return (c+1)/2


#for discriminator, input is a 64*64 RGB-face and tags[0...26]
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.tagfc = nn.Sequential(
			nn.Linear(26,64),
			nn.ReLU()
			)
		self.cmap = nn.Sequential(
			nn.Conv2d(3,64,4,stride=2,padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(64),
			nn.Conv2d(64,128,4,stride=2,padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(128),
			nn.Conv2d(128,256,4,stride=2,padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(256),
			nn.Conv2d(256,1,4,stride=1,padding=0),
			)
		self.conv = nn.Sequential(
			nn.Conv2d(65,16,3),
			nn.LeakyReLU(),
			nn.BatchNorm2d(16),
			nn.Conv2d(16,4,3),
			)
		self.fcm = nn.Sequential(
			nn.Linear(89,1),
			)
	def forward(self, x, tags):
		x = self.cmap(x)
		tags = self.tagfc(tags)
		x = x.view(x.shape[0],1*5*5)
		x = torch.cat((x,tags),1)
		x = self.fcm(x)
		x = F.sigmoid(x)
		return x



def getImg(array):
	array=array.reshape(3,64*64)
	img=[]
	for i in range(64*64):
		img.append([array[0][i],array[1][i],array[2][i]])
	img=np.array(img)
	img=np.clip(img,0,1)
	return img.reshape(64,64,3)




tagset = getTags()

def teston(epoch):
	generator = Generator()
	discrimitor = Discriminator()
	generator.load_state_dict(torch.load('./GModel/'+str(epoch)+'DCCGs.pth',map_location='cpu'))
	features = tagset[0].copy()
	features.extend(tagset[1])
	features.extend(tagset[2])
	gvec,gtag = GSampler(25)
	gout = generator(gvec)
	gout = gout.detach().numpy()
	gout = gout.reshape(25,3,64,64)
	i = 0
	for idata in gout:
		i = i + 1
		img = getImg(idata)
		plt.subplot(5,5,i)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img)
	plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
	plt.savefig('./sequence/'+str(epoch)+'epoch.jpg')


epochset=[80,180,260,280,320,420,500,640,700,760,800,840,900,960,1020,1500]
for epoch in epochset:
	teston(epoch)
