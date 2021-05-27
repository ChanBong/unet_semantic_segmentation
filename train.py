import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import UNet
from dataset import HelenDataset
from torch.utils.data import DataLoader
from utils import twersky, IOU, intLabels, label_grid

# set dir_path as the path to the unet directory
dir_path='./'

# Train on GPU, if available
if torch.cuda.is_available():
  device=torch.device('cuda')
else:
  device=torch.device('cpu')

# Init model
BATCH_SIZE=12
unet_model=UNet(in_channels=3, n_classes=11, batch_size=BATCH_SIZE)
unet_model.to(device)

# Load Pre-trained weights
print(os.path.join(dir_path,'model_weights.pth'))
unet_model.load_state_dict(torch.load(os.path.join(dir_path,'model_weights.pth'), map_location=device))

# Prepare Dataset
train_label_path= os.path.join(dir_path, "data/train/labels")
train_image_path= os.path.join(dir_path, "data/train/images")
training_data = HelenDataset(label_dir=train_label_path, img_dir=train_image_path)

test_label_path= os.path.join(dir_path, "data/test/labels")
test_image_path= os.path.join(dir_path, "data/test/images")
test_data = HelenDataset(label_dir=test_label_path, img_dir=test_image_path)

train_dataloader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True)

train_images, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_images.size()}")
print(f"Labels batch shape: {train_labels.size()}")


##Tensorboard visualization

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/exp_1')

images, labels = next(iter(train_dataloader))
img_grid = torchvision.utils.make_grid(images, nrow=4)
img_grig = np.array(img_grid)
img_grid = np.transpose(img_grid,(1,2,0))
plt.imshow(img_grid)

images, labels = next(iter(train_dataloader))
img_grid = torchvision.utils.make_grid(images, nrow=4)
writer.add_image('images', img_grid)

writer.add_graph(unet_model, images.to(device))

classes =['bg','face','lb','rb','le','re','nose','ulip','imouth','llip','hair']

optimizer = optim.Adam(unet_model.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, min_lr=0.000001, eps=1e-08, verbose=True)

def train(num_epochs, initial_loss, plot_gradients=False):
    start=0
    end=200
    running_loss = initial_loss
    for epochs in tqdm(range(num_epochs)):
      for iteration in range(start,end):
          print(iteration)
          images, labels = next(iter(train_dataloader))
          images=images.to(device)
          labels=labels.to(device)

          preds = unet_model(images)
          loss = twersky(preds, labels)

          optimizer.zero_grad() #set gradients 0
          loss.backward() #calculate gradients

          nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=0.01, norm_type=2.0)
          optimizer.step() #update weights

          running_loss = running_loss*0.9 + loss.item()*0.1

          if plot_gradients and j%200==1:
            for tag, parm in unet_model.named_parameters():
              writer.add_histogram(tag, parm.grad.data.cpu().numpy(), iteration//200)
          
          if iteration%10==1:
            print('  loss==> '+str(running_loss))

          if iteration % 50 == 1:
            scheduler.step(running_loss) #update lr
            writer.add_scalar('training loss', loss.item(), iteration)

            test_images, test_labels = next(iter(test_dataloader))
            test_images=test_images.to(device)
            test_labels=test_labels.to(device)

            unet_model.eval()
            preds = unet_model(test_images)
            iou = IOU(preds, test_labels)
            loss = twersky(preds, test_labels)
            writer.add_scalar('test loss', loss.item(), iteration)

            for i in range(11):
              writer.add_scalar('training iou class '+classes[i], iou[i], iteration)
            unet_model.train(True)
            torch.cuda.empty_cache()

          del images
          del labels

train(num_epochs=40,initial_loss=BATCH_SIZE*11)
torch.cuda.empty_cache()


# PLOT RESULTS
test_imgpath=os.path.join(dir_path,'data/test/images')
test_labelpath=os.path.join(dir_path,'data/test/labels')
test_predpath=os.path.join(dir_path,'data/test/preds')

test_imglist=os.listdir(os.path.join(dir_path,'data/test/images'))
test_labellist=os.listdir(os.path.join(dir_path,'data/test/labels'))

test_imglist.sort()
test_labellist.sort()

for i in range(len(test_labellist)):
  img=Image.open(os.path.join(test_imgpath, test_imglist[i]))
  img=np.array(img)
  img=np.transpose(img,(2,0,1))
  img_tensor=torch.tensor(img,dtype=torch.float32)
  img_tensor=img_tensor.unsqueeze(0)

  label=Image.open(os.path.join(test_labelpath, test_labellist[i]))
  label=np.array(label)
  label_tensor=torch.tensor(label,dtype=torch.float32)
  label_tensor=label_tensor.unsqueeze(0)

  pred=unet_model(img_tensor.to(device))

  pred=np.array(pred.clone().detach().cpu())
  pred=intLabels(pred[0])
  pred=pred.astype(np.uint8)

  pred=Image.fromarray(pred)
  pred=pred.save(os.path.join(test_predpath, test_labellist[i]))


images, labels = next(iter(test_dataloader))
preds = unet_model(images.to(device))
x=torch.tensor(preds.clone().detach(),device='cpu')

preds=np.array(preds.clone().detach().cpu())
labels=np.array(labels.clone().detach().cpu())

pred_grid=label_grid(preds)
label_grid_object=label_grid(labels)

plt.imshow(pred_grid)
plt.imshow(label_grid_object)

