import numpy as np
from PIL import Image
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
train_batch_size = 200
test_batch_size = 200
num_epochs = 5000
d_steps = 3
d_learning_rate= 1e-4
g_steps = 1
g_learning_rate = 5e-4
optim_betas = (0.5, 0.9)
img_shape = [1,32,32]
input_size = img_shape[0]*img_shape[1]*img_shape[2]
#input_size = 128

sample_dir = './samples/'


def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def main():
  trainset = MnistDataset('./data/mnist.pkl','train')
  testset = MnistDataset('./data/mnist.pkl','test')

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
  #D = Disc(img_shape, block_num=3 )
  D = WGAN_D(32, 1, 64,3)
  G = WGAN_G()
  G = iRevGener(img_shape,block_num=10)
  print(G)
  print(D)
  #criterion = thnn.BCEWithLogitsLoss()
  if use_cuda:
        D,G = D.cuda(),G.cuda()
        D = thnn.DataParallel(D, device_ids=(0,3))

        G= thnn.DataParallel(G, device_ids=(0,3))
        #range(torch.cuda.device_count()))
        cudnn.benchmark = True

  
  d_optimizer = thopt.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
  g_optimizer = thopt.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
  D.train()
  G.train()
  for epoch in range(num_epochs):
    print('======epoch %d ======'%epoch)
    d_index = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
      for p in D.parameters():  # reset requires_grad
        p.requires_grad = True
      #for p in D.parameters():
      #  p.data.clamp_(-0.01,0.01)
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
      #d_real_error.backward() # compute/store gradients, but don't change params
      
      d_gen_input = thV(torch.Tensor(np.random.normal(0,1,(batch_size,input_size,1,1))))
      if use_cuda:
        d_gen_input=d_gen_input.cuda()
      d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
      d_fake_decision = D(d_fake_data)

      d_fake_error = torch.mean(d_fake_decision)  # zeros = fake
      #d_fake_error.backward()
      error = d_fake_error+d_real_error
      error.backward()
      d_optimizer.step()
      for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)

      print('batch_idx %d       '%batch_idx+'d_real_error = %f    d_fake_error = %f'%(-d_real_error,d_fake_error),end='\r')
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
          
    index_list = np.random.randint(0,trainset.__len__(),size=(10,1)).tolist()
    for i,index in enumerate(index_list):
      samp = testset.__getitem__(index);
      im = np.uint8(samp[0][0].reshape((32,32))*255)
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
      im = g_fake_data.data[idx,:,:,:].view((32,32)).cpu().numpy()
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

if __name__ == '__main__':
   main()
