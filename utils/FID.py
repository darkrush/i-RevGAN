import numpy as np
import torch
import os
import inception
import pathlib
from scipy import linalg
from scipy.misc import imread
from scipy.misc import imresize 
from torch.autograd import Variable as thV
#import torchvision as tvs

def get_v3_model():
  v3 = inception.inception_v3(pretrained = True)
  return v3


def get_activations(images, model, batch_size=50, verbose=False):
  """Calculates the activations of the pool_3 layer for all images.
  Params:
  -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                   must lie between 0 and 256.
  -- sess        : current session
  -- batch_size  : the images numpy array is split into batches with batch size
                   batch_size. A reasonable batch size depends on the disposable hardware.
  -- verbose    : If set to True and parameter out_step is given, the number of calculated
                   batches is reported.
  Returns:
  -- A numpy array of dimension (num images, 2048) that contains the
     activations of the given tensor when feeding inception with the query tensor.
  """
  d0 = images.shape[0]
  if batch_size > d0:
      print("warning: batch size is bigger than the data size. setting batch size to data size")
      batch_size = d0
  n_batches = d0//batch_size
  n_used_imgs = n_batches*batch_size
  pred_arr = np.empty((n_used_imgs,2048))
  for i in range(n_batches):
      if verbose:
          print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
      start = i*batch_size
      end = start + batch_size
      batch = images[start:end]
      batch = thV(torch.Tensor(batch))
      pred = model(batch)
      pred = pred.view(batch_size,-1)
      pred_arr[start:end] = pred.data.cpu().numpy()
  if verbose:
      print(" done")
  return pred_arr





def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
      d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
      
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1 : Numpy array containing the activations of the pool_3 layer of the
       inception net ( like returned by the function 'get_predictions')
       for generated samples.
  -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
         on an representive data set.
  -- sigma1: The covariance matrix over activations of the pool_3 layer for
         generated samples.
  -- sigma2: The covariance matrix over activations of the pool_3 layer,
         precalcualted on an representive data set.
  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
  assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

  diff = mu1 - mu2

  # product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
    warnings.warn(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError("Imaginary component {}".format(m))
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  
  
  
def calculate_activation_statistics(images, model, batch_size=50, verbose=False):
  """Calculation of the statistics used by the FID.
  Params:
  -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                   must lie between 0 and 255.
  -- sess        : current session
  -- batch_size  : the images numpy array is split into batches with batch size
                   batch_size. A reasonable batch size depends on the available hardware.
  -- verbose     : If set to True and parameter out_step is given, the number of calculated
                   batches is reported.
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the incption model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the incption model.
  """
  act = get_activations(images, model, batch_size, verbose)
  mu = np.mean(act, axis=0)
  sigma = np.cov(act, rowvar=False)
  return mu, sigma

def _handle_path(path, model):
  if path.endswith('.npz'):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
  else:
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    img_array = []
    for fn in files:
      img = imresize(imread(str(fn)).astype(np.float32),(299,299))
      if len(img.shape) == 2:   
        img = np.dstack((img,img,img))  
      img_array.append(img)
    x = np.array(img_array).transpose((0,3,1,2))
    m, s = calculate_activation_statistics(x, model)
  return m, s
  
def calculate_fid_given_paths(paths):
  ''' Calculates the FID of two paths. '''

  for p in paths:
    if not os.path.exists(p):
      raise RuntimeError("Invalid path: %s" % p)
  model = get_v3_model()
  model.eval()
  m1, s1 = _handle_path(paths[0], model)
  m2, s2 = _handle_path(paths[1], model)
  fid_value = calculate_frechet_distance(m1, s1, m2, s2)
  return fid_value