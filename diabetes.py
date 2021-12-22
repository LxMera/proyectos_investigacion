import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import scipy.io as sio
import glob
import cv2
from sklearn import svm
from scipy import signal
from skimage.filters import threshold_otsu
from matplotlib.patches import Circle
from sklearn.cluster import KMeans
from scipy import stats
sns.set_theme()
root='/content/drive/MyDrive/TrainModels/'

pixv={'LCA': 1, 'LPA': 2, 'MCA': 4, 'MPA': 8} 

def random_plot(sizx, Subj, labe, Gend, Late, maxi, mini, temp, angi):
  rand=2*np.random.choice(int(sizx[0]/2))
  t1='_'.join([Subj[rand],str(labe[rand]),Gend[rand],Late[rand]])
  t2='_'.join([Subj[rand+1],str(labe[rand+1]),Gend[rand+1],Late[rand+1]])

  plt.figure(figsize=(14,7))
  plt.subplot(1,6,1)
  plt.imshow(temp[rand], cmap='nipy_spectral', vmin=mini, vmax=maxi)
  plt.title(t1)
  plt.axis('off')

  plt.subplot(1,6,2)
  plt.imshow(angi[rand])
  plt.title(t1)
  plt.axis('off')

  plt.subplot(1,6,3)
  plt.imshow(temp[rand+1], cmap='nipy_spectral', vmin=mini, vmax=maxi)
  plt.title(t2)
  plt.axis('off')

  plt.subplot(1,6,4)
  plt.imshow(angi[rand+1])
  plt.title(t2)
  plt.axis('off')

  print('Índice :', rand)

def geometric_center(ref, show=False):
  ima=ref.copy()  
  yp,xp=np.shape(ima)
  xc=np.sum(ima!=0, axis=0)@np.arange(xp)/np.sum(ima!=0)
  yc=np.sum(ima!=0, axis=1)@np.arange(yp)/np.sum(ima!=0)
  ima[:,int(xc)]=0
  ima[int(yc),:]=0
  if show:
    plt.imshow(ima, cmap='nipy_spectral', vmin=mini, vmax=maxi)
    plt.title('Centro geométrico')
    plt.axis('off')
  return xc,yc

def geometric_temp(ref, show=False):
  ima=ref.copy()
  yp,xp=np.shape(ima)
  xc=np.sum(ima, axis=0)@np.arange(xp)/np.sum(ima)
  yc=np.sum(ima, axis=1)@np.arange(yp)/np.sum(ima)
  ima[:,int(xc)]=maxi
  ima[int(yc),:]=maxi
  if show:
    plt.imshow(ima, cmap='nipy_spectral', vmin=mini, vmax=maxi)
    plt.title('Centro térmico')
    plt.axis('off')
  return xc,yc

