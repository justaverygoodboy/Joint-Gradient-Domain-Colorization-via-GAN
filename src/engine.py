import config
import torch
from tqdm import tqdm
from torch import nn
from lossfunc import PerceptualLoss
from lossfunc import GradientLoss
from lossfunc import GANLoss
import numpy as np
import utils

def train(train_loader, GAN_Model, netD, optG, optD, device, losses):
  batch = 0
  # def wgan_loss(prediction, real_or_not):
  #   if real_or_not:
  #     return -torch.mean(prediction.float())
  #   else:
  #     return torch.mean(prediction.float())
  # def gp_loss(y_pred, averaged_samples, gradient_penalty_weight):
  #   gradients = torch.autograd.grad(y_pred,averaged_samples,
  #                             grad_outputs=torch.ones(y_pred.size(), device=device),
  #                             create_graph=True, retain_graph=True, only_inputs=True)[0]
  #   gradients = gradients.view(gradients.size(0), -1)
  #   gradient_penalty = (((gradients+1e-16).norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
  #   return gradient_penalty
  adversarial_criterion = GANLoss()
  
  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  
      ########### add noise #################
      z = torch.randn((trainL.size(0),2,128,128),device=device) # change to normal distribution
      # trainL_3 = torch.tensor(np.tile(trainL.cpu(), [1,3,1,1]), device=device).float() #这里要不要把ab通道改成噪音
      trainL = torch.tensor(trainL, device=device).float()
      trainAB = torch.tensor(trainAB, device=device).float()
      trainL_3 = torch.cat([trainL,z],dim=1) # add noise to grayscale image for training
      ############ GAN MODEL ( Training Generator) ###################
      optG.zero_grad()
      predAB1,predAB2,predAB3,predAB4,predAB5, discpred = GAN_Model(trainL, trainL_3) 
      # D_G_z1 = discpred.mean().item()
      realLAB = torch.cat([trainL, trainAB], dim=1)
      predLAB1 = torch.cat([trainL, predAB1], dim=1)
      ############ G Loss ##################################
      # Loss_adver = adversarial_criterion(discpred, True)
      # Loss_WL = wgan_loss(discpred, True) 
      # Loss_Hinge = -discpred.mean()
      Loss_L1_1 = nn.L1Loss()(predAB1, trainAB) 
      Loss_L1_2 = nn.L1Loss()(predAB2, trainAB) 
      Loss_L1_3 = nn.L1Loss()(predAB3, trainAB) 
      Loss_L1_4 = nn.L1Loss()(predAB4, trainAB) 
      Loss_L1_5 = nn.L1Loss()(predAB5, trainAB) 
      Loss_L1 = Loss_L1_1+Loss_L1_2+Loss_L1_3+Loss_L1_4+Loss_L1_5 
      # Loss_Percp = PerceptualLoss()(predLAB,realLAB)
      # Loss_Gradient = GradientLoss()(predAB,trainAB)
      # Loss_G = Loss_adver*0.01 + Loss_L1 + Loss_Gradient*0.01 + Loss_Percp*0.00001
      Loss_G = Loss_L1
      Loss_G.backward()
      optG.step() 
      losses['G_losses'].append(Loss_G.item())
      losses['EPOCH_G_losses'].append(Loss_G.item())
      ############### Discriminator Training #########################
      # for param in netD.parameters(): # 将D的设置为可BP
      #   param.requires_grad = True
      # optD.zero_grad()

      # ###### hinge loss ######
      # # d_real
      # # d_out_real = netD(realLAB)
      # # d_loss_real = nn.ReLU()(1.0-d_out_real).mean()
      # # ## d_fake
      # # d_out_fake = netD(predLAB.detach())
      # # d_loss_fake = nn.ReLU()(1.0+d_out_fake).mean()
      # # d_loss = d_loss_real + d_loss_fake
      # # d_loss.backward()

      # ###### Unet bce loss ########
      # d_out_real = netD(realLAB)
      # d_loss_real = adversarial_criterion(d_out_real, True)
      # d_out_fake = netD(predLAB.detach())
      # d_loss_fake = adversarial_criterion(d_out_fake, False)
      # d_loss = (d_loss_real + d_loss_fake)/2
      # d_loss.backward()

      # optD.step()
      # losses['D_losses'].append(d_loss.item())
      # losses['EPOCH_D_losses'].append(d_loss.item())

      #Output training stats
      # if batch % 100 == 0: #原本是100
      #   print('L1: %.8f | Loss_Perc:%.8f |Loss_Grad:%.8f | Loss_D: %.8f | Loss_G: %.8f | D(x): %.8f | D(G(z)): %.8f |'
      #       % (Loss_L1.item(),Loss_Percp.item(),Loss_Gradient.item(),d_loss.item(), Loss_G.item(), d_loss_real, d_loss_fake))

      