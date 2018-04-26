import numpy as np
from PIL import Image
import os
import torch
import torch.nn as thnn
import torch.optim as thopt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable as thV

from models.iRevGan import iRevGener
from models.iRevGan import Gener
from models.iRevGan import Disc
from models.iRevGan import DCGAN_D
from models.wgan import WGAN_D
from models.wgan import WGAN_G
from models.mnist import MnistDataset

use_cuda = True
train_batch_size = 256
test_batch_size = 256
num_epochs = 2000
d_steps = 5
d_learning_rate= 2e-4
g_steps = 1
g_learning_rate = 4e-4
optim_betas = (0.5, 0.9)
CLIP_BOUND = 0.01
GAN_out_shape = [1,64,64]
img_shape = [1,32,32]
input_size = GAN_out_shape[0]*GAN_out_shape[1]*GAN_out_shape[2]
#input_size = 128
ROOT_PATH = './WGAN-big/'
LOG_PATH = ROOT_PATH+'logs/'
SNAP_PATH = ROOT_PATH+'snapshot/'
sample_dir = ROOT_PATH+'samples/'

def main():
  trainset = MnistDataset('./data/mnist.pkl',GAN_out_shape,'train')
  testset = MnistDataset('./data/mnist.pkl',GAN_out_shape,'test')

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
  #D = Disc(img_shape, block_num=3 )
  D = WGAN_D(32, 1, 64,3,GAN_out_shape)
  #G = WGAN_G()
  G = iRevGener(GAN_out_shape,block_num=5)
  if use_cuda:
    D,G = D.cuda(),G.cuda()
    D = thnn.DataParallel(D, device_ids=(0,3))
    G= thnn.DataParallel(G, device_ids=(0,3))
    cudnn.benchmark = True

  if os.path.exists(SNAP_PATH+'D.pkl'):
    D.load_state_dict(torch.load(SNAP_PATH+'D.pkl'))
    print('!!!!!Loaded D parameter!!!!!')
  if os.path.exists(SNAP_PATH+'G.pkl'):
    G.load_state_dict(torch.load(SNAP_PATH+'G.pkl'))
    print('!!!!!Loaded G parameter!!!!!')
  print(G)
  print(D)
  #criterion = thnn.BCEWithLogitsLoss()

  
  d_optimizer = thopt.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
  g_optimizer = thopt.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
  D.train()
  G.train()
  with open(LOG_PATH+'log.txt', 'w') as f:
    print('') 
  for epoch in range(num_epochs):
    print('\n======epoch %d ======'%epoch)
    d_index = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
      for p in D.parameters():  # reset requires_grad
        p.requires_grad = True
      batch_size = targets.size()[0]
      onesV = thV(torch.ones((batch_size,1)))
      zerosV = thV(torch.zeros((batch_size,1)))
      if use_cuda:
        onesV = onesV.cuda()
        zerosV = zerosV.cuda()
      D.zero_grad()
      if use_cuda:
        inputs = inputs.cuda()
      d_real_data = thV(inputs)
      if use_cuda:
        d_real_data=d_real_data.cuda()
      d_real_decision = D(d_real_data)
      d_real_error = -torch.mean(d_real_decision)  # ones = true
      
      d_gen_input = thV(torch.Tensor(np.random.normal(0,1,(batch_size,input_size,1,1))))
      if use_cuda:
        d_gen_input=d_gen_input.cuda()
      d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
      d_fake_decision = D(d_fake_data)

      d_fake_error = torch.mean(d_fake_decision)  # zeros = fake
      error = d_fake_error+d_real_error
      error.backward()
      d_optimizer.step()
      for p in D.parameters():
        p.data.clamp_(-CLIP_BOUND, CLIP_BOUND)
      with open(LOG_PATH+'log.txt', 'a+') as f:
        print('epoch %d '%epoch+ 'batch_idx %d       '%batch_idx+'d_real_error = %f    d_fake_error = %f'%(-d_real_error,d_fake_error),end='\n',file = f)
      print('epoch %d '%epoch+ 'batch_idx %d       '%batch_idx+'d_real_error = %f    d_fake_error = %f'%(-d_real_error,d_fake_error),end='\r')
      d_index=d_index+1
      if d_index%d_steps == 0:
        for p in D.parameters():
          p.requires_grad = False  # to avoid computation
        for g_index in range(g_steps):
          #print('g_index %d'%g_index)
          G.zero_grad()
          gen_input = thV(torch.Tensor(np.random.normal(0,1,(batch_size,input_size,1,1))))
          if use_cuda:
            gen_input=gen_input.cuda()
          g_fake_data = G(gen_input)
          dg_fake_decision = D(g_fake_data)
          g_error = -torch.mean(dg_fake_decision)  # we want to fool, so pretend it's all genuine

          g_error.backward()
          g_optimizer.step()  # Only optimizes G's parameters
          
    index_list = np.random.randint(0,testset.__len__(),size=(10,1)).tolist()
    for i,index in enumerate(index_list):
      samp = testset.__getitem__(index);
      im = np.uint8(samp[0][0].reshape((GAN_out_shape[1],GAN_out_shape[2]))*255)
      if i == 0:
        img_concat = im
      else:
        img_concat = np.column_stack((img_concat,im))
    real_img_concat = img_concat
    gen_input = thV(torch.Tensor(np.random.normal(0,1,(10,input_size,1,1))))
    if use_cuda:
      gen_input=gen_input.cuda()
    g_fake_data = G(gen_input)
    for idx in range(10):
      im = g_fake_data.data[idx,:,:,:].view((GAN_out_shape[1],GAN_out_shape[2])).cpu().numpy()
      im = np.uint8(im*255)
      if idx == 0:
        img_concat = im
      else:
        img_concat = np.column_stack((img_concat,im))
    fake_img_concat = img_concat
    img = np.row_stack((real_img_concat, fake_img_concat))
    im = Image.fromarray(img)
#    im.show()
    im.save(sample_dir+'%d.jpg'%(epoch))
    torch.save(G.state_dict(), SNAP_PATH+'G.pkl')
    torch.save(D.state_dict(), SNAP_PATH+'D.pkl')


if __name__ == '__main__':
   main()
