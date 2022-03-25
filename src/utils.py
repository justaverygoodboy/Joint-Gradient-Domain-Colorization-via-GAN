import os
import cv2
import torch
import config
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def preprocess(imgs):
  try:
    imgs = imgs.detach().numpy()
  except:
    pass
  imgs = imgs * 255
  imgs[imgs>255] = 255
  imgs[imgs<0] = 0 
  return imgs.astype(np.uint8) # torch.unit8
    
def reconstruct_no(batchX, predictedY):
    batchX = batchX.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,1) 
    predictedY = predictedY.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,2)
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2RGB)
    return result

def imag_gird(orig, batchL, preds, epoch,idx):
  _ , ax = plt.subplots(1,3, figsize=(15,15))
  ax[0].imshow(orig)
  ax[0].set_title('Original Image')
  ax[1].imshow(np.tile(batchL,(1,1,3)))
  ax[1].set_title('L Image with Channels reapeated(Input)') 
  ax[2].imshow(preds)
  ax[2].set_title('Pred Image')
  plt.savefig(f'../result/sample_preds_{epoch}_{idx}')
  plt.close()

def show_test_img(L,AB,epoch,idx): #单纯的保存图片，输入tensor L，AB
  AB = AB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
  L = L.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,1))
  img = reconstruct_no(preprocess(L), preprocess(AB))
  plt.imshow(img)
  plt.savefig(f'test_{epoch}_{idx}')

def show_single_channel(x,name): #保存单通道图片 输入tensor和保存name
  x = x.cpu().numpy().reshape((x.shape[2],x.shape[2],1))
  x = preprocess(x)
  plt.imshow(x)
  plt.savefig(f'test_{name}')

def plot_some(type,test_data, model, device, epoch):
  with torch.no_grad():
    if (type=="AE"):
      dataLen = len(test_data)
      # indexes = [0, 2, 9]
      for idx in range(dataLen):
        transf = transforms.ToTensor()
        batchL, realAB, filename = test_data[idx]
        filepath = config.TRAIN_DIR+filename
        batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
        realAB = realAB.reshape(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE)
        batchL = torch.tensor(batchL).to(device).float()
        realAB = torch.tensor(realAB).to(device).float()
        model.eval()
        realLAB = torch.cat([batchL, realAB], dim=1) #真实的图像
        recLAB = model(realLAB)
        show_test_img(recLAB[:,0:1,:,:,],recLAB[:,1:3,:,:,],epoch,idx)
    else:
      indexes = [0, 2, 9]
      for idx in indexes:
        batchL, realAB, filename = test_data[idx]
        filepath = config.TRAIN_DIR+filename
        batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
        realAB = realAB.reshape(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE)
        batchL_3 = torch.tensor(np.tile(batchL, [1, 3, 1, 1]))
        batchL_3 = batchL_3.to(device).float()
        batchL = torch.tensor(batchL).to(device).float()
        realAB = torch.tensor(realAB).to(device).float()
        model.eval()
        batch_predAB = model(batchL_3)
        batch_predAB = batch_predAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
        batchL = batchL.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,1))
        realAB = realAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
        orig = cv2.imread(filepath)
        orig = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (config.IMAGE_SIZE,config.IMAGE_SIZE))
        preds = reconstruct_no(preprocess(batchL), preprocess(batch_predAB))
        imag_gird(orig, batchL, preds, epoch-1,idx)

def create_checkpoint(type,epoch, netG, optG, netD, optD, max_checkpoint, save_path=config.GAN_CHECKPOINT_DIR):
  print('Saving Model and Optimizer weights.....')
  if (type=="AE"):
    checkpoint = {
          'epoch' : epoch,
          'generator_state_dict' :netG.state_dict(),
          'generator_optimizer': optG.state_dict(),
          'discriminator_state_dict': netD.state_dict(),
          'discriminator_optimizer': optD.state_dict()
      }
  else:
    checkpoint = {
        'epoch' : epoch,
        'AE_state_dict' :netG.state_dict(),
        'AE_optimizer': optG.state_dict(),
    }   
  torch.save(checkpoint, f'{save_path}{epoch}_checkpoint.pt')
  print('Weights Saved !!')
  del checkpoint
  files = glob.glob(os.path.expanduser(f"{save_path}*"))
  sorted_files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
  if len(sorted_files) > max_checkpoint:
    os.remove(sorted_files[-1])

def load_checkpoint(type,checkpoint_directory, netG, optG, netD, optD, device):
  if (type=="AE"): # ae中是 load_checkpoint(type,dir,net,opt,None,None,device)
    load_from_checkpoint = False
    files = glob.glob(os.path.expanduser(f"{checkpoint_directory}*"))
    for file in files:
        if file.endswith('.pt'):
            load_from_checkpoint=True
            break
    if load_from_checkpoint:
        print('Loading Pretrained AE Model and optimizer states from checkpoint....')
        sorted_files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
        checkpoint = torch.load(f'{sorted_files[0]}')
        epoch_checkpoint = checkpoint['epoch'] + 1
        netG.load_state_dict(checkpoint['AE_state_dict'])
        netG.to(device)
        optG.load_state_dict(checkpoint['AE_optimizer'])
        print('Loaded States !!!')
        print(f'It looks like the this states belong to epoch {epoch_checkpoint-1}.')
        return netG, optG, epoch_checkpoint
    else:
        print('There are no checkpoints.')
        epoch_checkpoint = 1
        return netG, optG, epoch_checkpoint
  else:
    load_from_checkpoint = False
    files = glob.glob(os.path.expanduser(f"{checkpoint_directory}*"))
    for file in files:
        if file.endswith('.pt'):
            load_from_checkpoint=True
            break
    if load_from_checkpoint:
        print('Loading Model and optimizer states from checkpoint....')
        sorted_files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
        checkpoint = torch.load(f'{sorted_files[0]}')
        epoch_checkpoint = checkpoint['epoch'] + 1
        netG.load_state_dict(checkpoint['generator_state_dict'])
        netG.to(device)
        optG.load_state_dict(checkpoint['generator_optimizer'])
        netD.load_state_dict(checkpoint['discriminator_state_dict'])
        netD.to(device)
        optD.load_state_dict(checkpoint['discriminator_optimizer'])
        print('Loaded States !!!')
        print(f'It looks like the this states belong to epoch {epoch_checkpoint-1}.')
        print(f'so the model will train for {config.NUM_EPOCHS - (epoch_checkpoint-1)} more epochs.')
        print(f'If you want to train for more epochs, change the "NUM_EPOCHS" in config.py !!')
        return netG, optG, netD, optD, epoch_checkpoint
    else:
        print('There are no checkpoints in the mentioned directoy, the Model will train from scratch.')
        epoch_checkpoint = 1
        return netG, optG, netD, optD, epoch_checkpoint
    
def plot_gan_loss(G_losses, D_losses,epoch):
  plt.figure(figsize=(10,5))
  plt.title(f"Generator and Discriminator Loss During Training ")
  plt.plot(G_losses,label="G")
  plt.plot(D_losses,label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(f'GANLOSS{epoch}.png',figsize=(15,10))
