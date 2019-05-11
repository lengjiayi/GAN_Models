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

from google.colab import drive
drive.mount('/content/gdrive')


start_epoch=260


datasize = 13000

learning_rate = 0.0002
epoch_num = 300
batch_size = 200
D_step = 1
G_step = 1
EarlyStop = True


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
		img=mping.imread('./selected64/'+str(i)+'.jpg')
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
	return np.array(faces).astype(float)/128-1,np.array(facetags)


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
#	return xs,tags
	return xs.cuda(),tags.cuda()

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
#	return tags
	return tags.cuda()


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
		return c


#for discriminator, input is a 64*64 RGB-face and tags[0...26]
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
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
			nn.Linear(25,1),
			)
	def forward(self, x):
		x = self.cmap(x)
		x = x.view(x.shape[0],1*5*5)
		x = self.fcm(x)
		x = F.sigmoid(x)
		return x




criterion = nn.BCEWithLogitsLoss()
criterion.cuda()


def DLossfun(dreal,dfake):
#	onewithep = torch.ones(drealm.shape[0],1)-torch.rand(drealm.shape[0],1)*0.05
#	onewithep = torch.ones(dreal.shape[0],1).cuda()
#	onewithep = (torch.ones(dreal.shape[0],1)-torch.rand(dreal.shape[0],1)*0.1)
#	zerowithep = torch.zeros(dfake.shape[0],1)
	onewithep = (torch.ones(dreal.shape[0],1)-torch.rand(dreal.shape[0],1)*0.1).cuda()
	zerowithep = (torch.zeros(dreal.shape[0],1)+torch.rand(dreal.shape[0],1)*0.1).cuda()
	realloss = criterion(dreal,onewithep)
	fakeloss = (1-criterion(dfake,onewithep))
	return (realloss+fakeloss)/2

def GLossfun(dout):
	gloss = criterion(dout,torch.ones(dout.shape[0],1).cuda())
#	gloss = criterion(dout,torch.ones(dout.shape[0],1))
	return gloss


def train(epoch_num):
	for epoch in range(epoch_num):
		#firstly, train D
		for d_epoch in range(D_step):
			i = 0
			for face,tag in dataloader:
				face = face
				tag = tag
				face = face.cuda()
				tag = tag.cuda()
				i += 1
				Doptim.zero_grad()
				#train on real data
				drealout = discrimitor(face)
				#train on fake data
				gvec,gtag = GSampler(face.shape[0])
				gout = generator(gvec).detach()
				dfakeout = discrimitor(gout)
				#train on mislabel
				d_loss = DLossfun(drealout,dfakeout)
				dacc_r = (sum(drealout)/drealout.shape[0]).item()
				dacc_f = (1-sum(dfakeout)/dfakeout.shape[0]).item()
				if EarlyStop and dacc_f>1-0.0001:
					print('\tearlyD_epoch:{}/{}: \tAcc:{:.4f}, {:.4f}'.format(d_epoch, D_step, dacc_r,dacc_f))
					break
				d_loss.backward()
				Doptim.step()
			print('\tD_epoch:{}/{}: \tAcc:{:.4f}, {:.4f}'.format(d_epoch, D_step, dacc_r,dacc_f))
		#secondly, train G
		for g_epoch in range(G_step):
			i = 0
			for batch in range(int(round(datasize/batch_size))):
				i += 1
				Goptim.zero_grad()
				gvec,gtag = GSampler(batch_size)
				gout = generator(gvec)
				dout = discrimitor(gout)				
				g_loss = GLossfun(dout)
				gacc = sum(dout)/dout.shape[0]
				if EarlyStop and gacc.item()>1-0.0001:
					print('\tearlyG_epoch:{}/{}: GAcc:{:.4f}\t'.format(g_epoch, G_step, gacc.item()))
					break
				g_loss.backward()
				Goptim.step()
			print('\tG_epoch:{}/{}: GAcc:{:.4f}\t'.format(g_epoch, G_step, gacc.item()))
		if epoch%20==0 and epoch!=0:
			print('savepoint')
			torch.save(generator.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch+start_epoch)+'DCCGs.pth')
			torch.save(discrimitor.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch+start_epoch)+'DCCDs.pth')
			torch.save(Goptim.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch+start_epoch)+'DCCGsops.pth')
			torch.save(Doptim.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch+start_epoch)+'DCCDsops.pth')
		print('\nepoch:{}/{}'.format(start_epoch+epoch, start_epoch+epoch_num))
	


tagset = getTags()
faceset,facetagset = getfaces()

generator = Generator()
generator.cuda()
discrimitor = Discriminator()
discrimitor.cuda()
generator.load_state_dict(torch.load('./gdrive/My Drive/GAN/'+str(start_epoch)+'DCCGs.pth'))
discrimitor.load_state_dict(torch.load('./gdrive/My Drive/GAN/'+str(start_epoch)+'DCCDs.pth'))


traindata=TensorDataset(torch.from_numpy(faceset).float(),torch.from_numpy(facetagset).float())
dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)

Goptim = optim.Adam(generator.parameters(),lr=learning_rate)
Doptim = optim.Adam(discrimitor.parameters(),lr=learning_rate)
Goptim.load_state_dict(torch.load('./gdrive/My Drive/GAN/'+str(start_epoch)+'DCCGsops.pth'))
Doptim.load_state_dict(torch.load('./gdrive/My Drive/GAN/'+str(start_epoch)+'DCCDsops.pth'))

train(epoch_num)

torch.save(generator.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch_num+start_epoch)+'DCCGs.pth')
torch.save(discrimitor.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch_num+start_epoch)+'DCCDs.pth')
torch.save(Goptim.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch_num+start_epoch)+'DCCGsops.pth')
torch.save(Doptim.state_dict(),'./gdrive/My Drive/GAN/'+str(epoch_num+start_epoch)+'DCCDsops.pth')
