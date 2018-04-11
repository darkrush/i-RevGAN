import numpy
import pickle
import torch.utils

class MnistDataset(torch.utils.data.Dataset):
  def __init__(self,filepath,set):
    self.filepath = filepath
    self.set = set
    data = pickle.load(open(filepath,'rb'),encoding='iso-8859-1')
    if self.set == 'train':
      self.data,self.target = data[0]
    elif set == 'test':
      self.data,self.target = data[1]
    elif set == 'dev':
      self.data,self.target = data[2]
  def __getitem__(self, index):
    wdata = numpy.concatenate((self.data[index].reshape(1,28,28),self.data[index].reshape(1,28,28)),0)
    
    wdata = numpy.pad(wdata,((0,0),(2,2),(2,2)),'constant', constant_values=(0,0))
    return wdata,self.target[index]
  def __len__(self):
    if set == 'train':
      return 50000
    else:
      return 10000
    return 0