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
from models.wgan import WGAN_G
from models.mnist import MnistDataset

use_cuda = True
train_batch_size = 200
test_batch_size = 200
num_epochs = 2000
d_steps = 3
d_learning_rate= 1e-4
g_steps = 1
g_learning_rate = 2e-4
optim_betas = (0.5, 0.9)
img_shape = [1,32,32]
input_size = img_shape[0]*img_shape[1]*img_shape[2]
#input_size = 128
SNAP_PATH = './snapshot/WGAN/'
sample_dir = './samples/WGAN/'

def main():
  trainset = MnistDataset('./data/mnist.pkl','train')
  testset = MnistDataset('./data/mnist.pkl','test')

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

  G = iRevGener(img_shape,block_num=10)
  if use_cuda:
    G = G.cuda()
    G= thnn.DataParallel(G, device_ids=(0,1,2,3))
    cudnn.benchmark = True

  if os.path.exists(SNAP_PATH+'G.pkl'):
    G.load_state_dict(torch.load(SNAP_PATH+'G.pkl'))
    print('!!!!!Loaded G parameter!!!!!')
  
  np.set_printoptions(threshold=np.nan)
  

  test_input = np.empty(shape=[0, input_size])
  for batch_idx, (inputs, targets) in enumerate(testloader):
    if use_cuda:
      inputs = inputs.cuda()
    d_real_data = thV(inputs)
    inv_input = G.module.inverse(d_real_data)
    batch_input = inv_input.data.contiguous().view(-1,input_size).cpu().numpy()
    test_input = np.concatenate((test_input, batch_input), axis=0)
  np.save('test_input.npy',test_input)

  
  train_input = np.empty(shape=[0, input_size])
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    if use_cuda:
      inputs = inputs.cuda()
    d_real_data = thV(inputs)
    inv_input = G.module.inverse(d_real_data)
    batch_input = inv_input.data.contiguous().view(-1,input_size).cpu().numpy()
    train_input = np.concatenate((train_input, batch_input), axis=0)
  np.save('train_input.npy',train_input)

if __name__ == '__main__':
   main()
