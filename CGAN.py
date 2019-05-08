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


datasize = 8000

learning_rate = 0.0002
epoch_num = 100
batch_size = 160
D_step = 2
G_step = 2


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
		img=mping.imread('./selected/'+str(i)+'.jpg')
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
	return tags.cuda()


#for generator, input is a norm-distribution vector x[0...100] and random tags[0...26]
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.fc = nn.Linear(106,4*4*128)
		self.remap = nn.Sequential(
			nn.ConvTranspose2d(128,64,5,stride=2,padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64,32,5,stride=2,padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(32,5,6,stride=2,padding=1),
			)
		self.fc2 = nn.Linear(5*40*40,3*40*40)

	def forward(self, x, tags):
		c = torch.cat((x,tags),1) #batch_size*126
		c = self.fc(c).cuda()
		c = c.view(c.shape[0],128,4,4)
		c = self.remap(c)
		c = c.view(c.shape[0],5*40*40)
		c = self.fc2(c)
		c = c.view(c.shape[0],3,40,40)
		return c


#for discriminator, input is a 64*64 RGB-face and tags[0...26]
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.tagfc = nn.Linear(26,9*9)
		self.cmap = nn.Sequential(
			nn.Conv2d(3,32,6,stride=2,padding=1),
			nn.LeakyReLU(),
			nn.BatchNorm2d(32),
			nn.Conv2d(32,64,5,stride=2,padding=1),
			)
		self.conv = nn.Sequential(
			nn.Conv2d(65,16,3),
			nn.LeakyReLU(),
			nn.BatchNorm2d(16),
			nn.Conv2d(16,4,3),
			)
		self.fcm = nn.Sequential(
			nn.Linear(4*5*5,1),
			)
	def forward(self, x, tags):
		x = self.cmap(x)
		tags = self.tagfc(tags)
		tags = tags.view(tags.shape[0],1,9,9)
		x = torch.cat((x,tags),1)
		x = self.conv(x)
		x = x.view(x.shape[0],4*5*5)
		x = self.fcm(x)
		x = F.sigmoid(x)
		return x



def getImg(array):
	array=array.reshape(3,40*40)
	img=[]
	for i in range(40*40):
		img.append([array[0][i],array[1][i],array[2][i]])
	img=np.array(img)
	img=np.clip(img,0,1)
	return img.reshape(40,40,3)


def vitest():
	gvec,gtag = GSampler(25)
	gout = generator(gvec,gtag)
	gout = gout.detach().numpy()
	gout = gout.reshape(25,3,40,40)
	i = 0
	for idata in gout:
		i = i + 1
		img = getImg(idata)
		plt.subplot(5,5,i)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(img)
	plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
	plt.show()




criterion = nn.BCEWithLogitsLoss()
criterion.cuda()


def DLossfun(dreal,dfake,dmiss):
#	onewithep = torch.ones(drealm.shape[0],1)-torch.rand(drealm.shape[0],1)*0.05
#	onewithep = torch.ones(dreal.shape[0],1).cuda()
	onewithep = (torch.ones(dreal.shape[0],1)-torch.rand(dreal.shape[0],1)*0.3).cuda()
	zerowithep = torch.zeros(dfake.shape[0],1).cuda()
	realloss = criterion(dreal,onewithep)
	fakeloss = (1-criterion(dfake,onewithep))
	missloss = (1-criterion(dmiss,onewithep))
	return realloss,fakeloss,missloss

def GLossfun(dout):
	gloss = criterion(dout,torch.ones(dout.shape[0],1).cuda())
	return gloss


def train(epoch_num):
	for epoch in range(epoch_num):
		#firstly, train D
		for d_epoch in range(D_step):
			d_totalloss = 0
			i = 0
			for face,tag in dataloader:
				face = face.cuda()
				tag = tag.cuda()
				i += 1
				Doptim.zero_grad()
				#train on real data
				drealout = discrimitor(face,tag)
				#train on fake data
				gvec,gtag = GSampler(face.shape[0])
				gout = generator(gvec,gtag).detach()
				dfakeout = discrimitor(gout,gtag)
				#train on mislabel
				mistag = DTagSampler(face.shape[0])
				dmissout = discrimitor(face,mistag)
				d_loss1,d_loss2,d_loss3 = DLossfun(drealout,dfakeout,dmissout)
				dacc_r = sum(drealout)/drealout.shape[0]
				dacc_f = 1-sum(dfakeout)/dfakeout.shape[0]
				dacc_m = 1-sum(dmissout)/dmissout.shape[0]
				if(dacc_r.item()>0.8) and (dacc_f.item()>0.8) and (dacc_m.item()>0.8):
					d_loss2.backward()
					break
				if dacc_r.item()<0.8:
					d_loss1.backward()
				if dacc_f.item()<0.8:
					d_loss2.backward()
				if dacc_m.item()<0.8:
					d_loss3.backward()
				Doptim.step()
				print('\tD_epoch:{}/{}: \tAcc:{:.4f}, {:.4f}, {:.4f}'.format(d_epoch, D_step, dacc_r.item(),dacc_f.item(),dacc_m.item()))
		#secondly, train G
		for g_epoch in range(G_step):
			g_totalloss = 0
			i = 0
			for batch in range(int(round(datasize/batch_size))):
				i += 1
				Goptim.zero_grad()
				gvec,gtag = GSampler(batch_size)
				gout = generator(gvec,gtag)
				dout = discrimitor(gout,gtag)				
				g_loss = GLossfun(dout)
				gacc = sum(dout)/dout.shape[0]
				if gacc.item()>0.8:
					print('\tG_epoch:{}/{}: G_batch_loss:{:.8f}\t\tGAcc:{:.4f}\t'.format(g_epoch, G_step, g_loss.item(),gacc.item()))
					break
				g_loss.backward()
				g_totalloss += float(g_loss)
				Goptim.step()
		print('\tG_epoch:{}/{}: G_batch_loss:{:.8f}\t\tGAcc:{:.4f}\t'.format(g_epoch, G_step, g_loss.item(),gacc.item()))
		if epoch%10==0:
#			vitest()
			print('savepoint')
			torch.save(generator.state_dict(),'./Gs.pth')
			torch.save(discrimitor.state_dict(),'./Ds.pth')
			torch.save(Goptim.state_dict(),'./Gsopt.pth')
			torch.save(Doptim.state_dict(),'./Dsopt.pth')
		print('\nepoch:{}/{}'.format(epoch, epoch_num))
	


tagset = getTags()
faceset,facetagset = getfaces()

generator = Generator()
generator.cuda()
discrimitor = Discriminator()
discrimitor.cuda()
generator.load_state_dict(torch.load('Gs.pth'))
discrimitor.load_state_dict(torch.load('Ds.pth'))


traindata=TensorDataset(torch.from_numpy(faceset).float(),torch.from_numpy(facetagset).float())
dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)

Goptim = optim.Adam(generator.parameters(),lr=learning_rate)
Doptim = optim.Adam(discrimitor.parameters(),lr=learning_rate)
Goptim.load_state_dict(torch.load('Gsopt.pth'))
Doptim.load_state_dict(torch.load('Dsopt.pth'))

train(epoch_num)
#vitest()

torch.save(generator.state_dict(),'./Gs.pth')
torch.save(discrimitor.state_dict(),'./Ds.pth')
torch.save(Goptim.state_dict(),'./Gsopt.pth')
torch.save(Doptim.state_dict(),'./Dsopt.pth')
