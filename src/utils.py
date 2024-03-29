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

def show_single_img(x,c,name): #保存单张图片【输入tensor图片，通道数，名字】
  x = x.cpu().numpy().reshape((x.shape[2],x.shape[2],c))
  x = preprocess(x)
  plt.imshow(x)
  plt.savefig(f'test_{name}')

def plot_some(type,test_data, model, device, epoch):
  with torch.no_grad():
    if (type=="AE"):
      # dataLen = len(test_data)
      indexes = [0,1,2,3,4,5,6,7,8,9]
      for idx in indexes:
        transf = transforms.ToTensor()
        batchL, realAB, filename = test_data[idx]
        filepath = config.TRAIN_DIR+filename
        batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
        realAB = realAB.reshape(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE)
        batchL = torch.tensor(batchL).to(device).float()
        # realAB = torch.tensor(realAB).to(device).float()
        noiseAB = torch.randn(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE,device=device)
        model.eval()
        realLAB = torch.cat([batchL, noiseAB], dim=1) #真实的图像
        recAB = model(realLAB)
        recLAB = torch.cat([batchL,recAB],dim=1)
        show_test_img(recLAB[:,0:1,:,:,],recLAB[:,1:3,:,:,],epoch,idx)
    elif(type=="test"):
      dataLen = len(test_data)
      for idx in range(dataLen):
        batchL, realAB, filename = test_data[idx]
        filepath = config.TEST_DIR+filename
        batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
        realAB = realAB.reshape(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE)
        batchL_3 = torch.tensor(np.tile(batchL, [1, 3, 1, 1]))

        # z = torch.randn((1,2,config.IMAGE_SIZE,config.IMAGE_SIZE),device=device) # change to normal distribution
        # batchL = torch.tensor(batchL, device=device).float()
        # batchL_3 = torch.cat([batchL,z],dim=1) # add noise to grayscale image for training
        batchL_3 = batchL_3.to(device).float()
        batchL = torch.tensor(batchL).to(device).float()
        realAB = torch.tensor(realAB).to(device).float()
        model.eval()
        batch_predAB,_,_ = model(batchL_3)
        batch_predAB = batch_predAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
        batchL = batchL.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,1))
        realAB = realAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
        orig = cv2.imread(filepath)
        orig = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (config.IMAGE_SIZE,config.IMAGE_SIZE))
        preds = reconstruct_no(preprocess(batchL), preprocess(batch_predAB))
        imag_gird(orig, batchL, preds, epoch-1,idx)
    else:
      indexes = [32, 12]
      for idx in indexes:
        batchL, realAB, filename = test_data[idx]
        filepath = config.TRAIN_DIR+filename
        batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
        realAB = realAB.reshape(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE)
        batchL_3 = torch.tensor(np.tile(batchL, [1, 3, 1, 1]))

        # z = torch.randn((1,2,config.IMAGE_SIZE,config.IMAGE_SIZE),device=device) # change to normal distribution
        # batchL = torch.tensor(batchL, device=device).float()
        # batchL_3 = torch.cat([batchL,z],dim=1) # add noise to grayscale image for training

        batchL_3 = batchL_3.to(device).float()
        batchL = torch.tensor(batchL).to(device).float()
        realAB = torch.tensor(realAB).to(device).float()
        model.eval()
        batch_predAB,_,_ = model(batchL_3)
        batch_predAB = batch_predAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
        batchL = batchL.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,1))
        realAB = realAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
        orig = cv2.imread(filepath)
        orig = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (config.IMAGE_SIZE,config.IMAGE_SIZE))
        preds = reconstruct_no(preprocess(batchL), preprocess(batch_predAB))
        imag_gird(orig, batchL, preds, epoch-1,idx)
# def save_att(test_data,model,device):
  
def save_test_images(test_data, model, device):
  with torch.no_grad():
    dataLen = len(test_data)
    for idx in range(dataLen):
      batchL,_,_ = test_data[idx]
      batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
      batchL_3 = torch.tensor(np.tile(batchL, [1, 3, 1, 1])).to(device).float()
      batchL = torch.tensor(batchL).to(device).float()
      model.eval()
      batch_predAB,_,_ = model(batchL_3)
      batch_predAB = batch_predAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
      batchL = batchL.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,1))
      batch_predAB = preprocess(batch_predAB)
      batchL = preprocess(batchL)
      batchL = batchL.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,1) 
      batch_predAB = batch_predAB.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,2)
      result = np.concatenate((batchL, batch_predAB), axis=2)
      result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
      cv2.imwrite(f'../sample/preds_{idx}.jpg',result)



def create_checkpoint(type,epoch, netG, optG, netDimg, optDimg,netDgrad,optDgrad, max_checkpoint, save_path=config.GAN_CHECKPOINT_DIR):
  print('Saving Model and Optimizer weights.....')
  if (type!="AE"):
    checkpoint = {
          'epoch' : epoch,
          'generator_state_dict' :netG.state_dict(),
          'generator_optimizer': optG.state_dict(),
          'discriminator_img_state_dict': netDimg.state_dict(),
          'discriminator_img_optimizer': optDimg.state_dict(),
          'discriminator_grad_state_dict': netDgrad.state_dict(),
          'discriminator_grad_optimizer': optDgrad.state_dict(),
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


def load_pretrained(checkpoint_directory, netG, device):
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
      netG.load_state_dict(checkpoint['generator_state_dict'],strict=False)
      netG.to(device)
      print('Loaded Pretrained Generator!!!')
      print(f'It looks like the this states belong to epoch {epoch_checkpoint-1}.')
      return netG, epoch_checkpoint
  else:
      assert False

def load_checkpoint(type,checkpoint_directory, netG, optG, netDimg, optDimg,netDgrad,optDgrad, device):
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
        netG.load_state_dict(checkpoint['AE_state_dict'],strict=False)
        netG.to(device)
        # optG.load_state_dict(checkpoint['AE_optimizer'])
        print('Loaded States !!!')
        print(f'It looks like the this states belong to epoch {epoch_checkpoint-1}.')
        return netG, epoch_checkpoint #optG, epoch_checkpoint
    else:
        print('There are no checkpoints.')
        epoch_checkpoint = 1
        return netG, optG, epoch_checkpoint
  elif(type=="test"):
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
        netG.load_state_dict(checkpoint['generator_state_dict'],strict=False)
        netG.to(device)
        print('Loaded States !!!')
        print(f'It looks like the this states belong to epoch {epoch_checkpoint-1}.')
        return netG, epoch_checkpoint
    else:
        print('There are no checkpoints in the mentioned directoy, the Model will train from scratch.')
        epoch_checkpoint = 1
        return netG, epoch_checkpoint
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
        netG.load_state_dict(checkpoint['generator_state_dict'],strict=False)
        netG.to(device)
        optG.load_state_dict(checkpoint['generator_optimizer'])
        netDimg.load_state_dict(checkpoint['discriminator_img_state_dict'],strict=False)
        netDimg.to(device)
        optDimg.load_state_dict(checkpoint['discriminator_img_optimizer'])
        netDgrad.load_state_dict(checkpoint['discriminator_grad_state_dict'],strict=False)
        netDgrad.to(device)
        optDgrad.load_state_dict(checkpoint['discriminator_grad_optimizer'])
        print('Loaded States !!!')
        print(f'It looks like the this states belong to epoch {epoch_checkpoint-1}.')
        print(f'so the model will train for {config.NUM_EPOCHS - (epoch_checkpoint-1)} more epochs.')
        print(f'If you want to train for more epochs, change the "NUM_EPOCHS" in config.py !!')
        return netG, optG, netDimg, optDimg,netDgrad,optDgrad, epoch_checkpoint
    else:
        print('There are no checkpoints in the mentioned directoy, the Model will train from scratch.')
        epoch_checkpoint = 1
        return netG, optG, netDimg, optDimg,netDgrad,optDgrad, epoch_checkpoint
    
# def plot_gan_loss(G_losses, D_losses,epoch):
#   plt.figure(figsize=(10,5))
#   plt.title(f"Generator and Discriminator Loss During Training ")
#   plt.plot(G_losses,label="G")
#   plt.plot(D_losses,label="D")
#   plt.xlabel("iterations")
#   plt.ylabel("Loss")
#   plt.legend()
#   plt.savefig(f'GANLOSS{epoch}.png',figsize=(15,10))

# def lab2xyz_tensor(lab):
#   # 传入的lab颜色值域0~1，真实的lab颜色值为l：0~100，a、b：-128~127
#   L = lab[:,0:1,:,:,]
#   A = lab[:,1:2,:,:,]
#   B = lab[:,2:3,:,:,]
#   # y = 
#   Xn=96.4221
#   Yn=100.0
#   Zn=82.5221
#   L = L*100
#   A = A*255-128.
#   B = B*255-128.
#   fy = (L+16.)/116
#   fx = (fy+A/500.)
#   fz = fy-B/200.
#   print(L.shape)
#   print(fy.shape)
#   assert False
#   delta = 6/29
#   if torch.det(fy)>delta:
#     Y = Yn*torch.mm(fy,torch.mm(fy,fy))
#   else:
#     Y = (fy-16/116)*3*delta*delta*Yn
#   if torch.det(fx)>delta:
#     X = Xn*torch.mm(fx,torch.mm(fx,fx))
#   else:
#     X = (fx-16/116)*3*delta*delta*Xn
#   if torch.det(fz)>delta:
#     Z = Zn*torch.mm(fz,torch.mm(fz,fz))
#   else:
#     Z = (fz-16/116)*3*delta*delta*Zn
#   XYZ = torch.cat([X,Y,Z],dim=1)
#   print(XYZ.shape)