import model
import config
import dataset
import utils
import engine

import torch
import warnings
warnings.filterwarnings('ignore')
import gc

def map_fn(index=None, flags=None):
  torch.set_default_tensor_type('torch.FloatTensor') #这里加了.cuda
  torch.manual_seed(1) #1234
  ######## 读取数据 ############
  train_data = dataset.DATA(config.TRAIN_DIR) 
  train_sampler = torch.utils.data.RandomSampler(train_data) # 这里应该是随机打乱数据
  ####### 用DataLoader读取进网络的数据 ###############
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=config.BATCH_SIZE,
      sampler=train_sampler,
      num_workers=6,
      drop_last=True,
      pin_memory=True)
  DEVICE = config.DEVICE
  ######### 定义模型 #################
  ## GAN网络
  netG = model.UNet_3Plus().float()
  netD = model.discriminator_model().float()

  ######################################
  netG = netG.to(DEVICE)
  netD = netD.to(DEVICE)
  ## 设置优化器参数
  optD = torch.optim.Adam(netD.parameters(), lr=4e-5, betas=(0.5, 0.999)) #论文里这里是2e-5
  optG = torch.optim.Adam(netG.parameters(), lr=2e-5, betas=(0.5, 0.999))
  ## Trains
  losses = {
      'G_losses' : [],
      'D_losses' : [],
      'EPOCH_G_losses' : [],
      'EPOCH_D_losses' : [],
      'G_losses_eval' : []
  }
  ## 读取保存的模型
  netG, optG, netD, optD, epoch_checkpoint = utils.load_checkpoint(config.CHECKPOINT_DIR, netG, optG, netD, optD, DEVICE)
  ## 将G和D组成GAN网络
  netGAN = model.GAN(netG, netD)

  for epoch in range(epoch_checkpoint,config.NUM_EPOCHS+1):
    print('\n')
    print('#'*8,f'EPOCH-{epoch}','#'*8)
    losses['EPOCH_G_losses'] = []
    losses['EPOCH_D_losses'] = []
    engine.train(train_loader, netGAN, netD, optG, optD, device=DEVICE, losses=losses)
    # utils.create_checkpoint(epoch, netG, optG, netD, optD, max_checkpoint=config.KEEP_CKPT, save_path = config.CHECKPOINT_DIR)
    utils.plot_some(train_data, netG, DEVICE, epoch)
    gc.collect() #这个貌似是垃圾回收
    ###### 这代码没测试吗 ############
  
if __name__=='__main__':
  map_fn()