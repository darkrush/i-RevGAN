import numpy as np
import torch as th
from torch.autograd import Variable as thV
from models.iRevGan import iRevGener
from models.iRevGan import Disc


def main():
  img_shape = [3,32,32]
  input_size = img_shape[0]*img_shape[1]*img_shape[2]

  model = iRevGener(img_shape,block_num=2)
  rand_number = np.random.normal(0,1,(1,input_size,1,1))
  rand_number = thV(th.FloatTensor(rand_number))
  result = model(rand_number)
  print(result.size())

if __name__ == '__main__':
   main()
