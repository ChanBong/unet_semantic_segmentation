import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def oneHotLabels(labels):
  labels = torch.from_numpy(labels)
  onehot=F.one_hot(labels.to(torch.int64),num_classes=11)
  return np.transpose(np.array(onehot),(2,0,1))

def intLabels(label):
  return np.argmax(label, axis=0)

def plotLabel(label):
  plt.imshow(label)
  
def plotImage(image):
  image = np.array(image)
  plt.imshow(np.transpose(image,(1,2,0)))

def twersky(y_pred, y_label):
    b=0.3
    #print(y_pred.size())
    x=y_pred*y_label
    intersection=torch.sum(x, dim=(2,3))
    union = b*torch.sum(y_label, dim=(2,3))+(1-b)*torch.sum(y_pred, dim=(2,3))
    loss = torch.sum(1-torch.div(intersection,union))
    return loss

def IOU(y_pred, y_label):
  x=y_pred*y_label
  intersection=torch.sum(x, dim=(2,3))
  union = torch.sum(y_label, dim=(2,3))+torch.sum(y_pred, dim=(2,3))+0.00001
  IOU_score = torch.sum(torch.div(intersection,union-intersection), dim=(0))
  return IOU_score


def label_grid(label):
  grid=intLabels(label[0])
  for i in range(1,3,1):
    grid=np.append(grid, intLabels(label[i]), axis=1)
  return grid
