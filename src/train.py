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
  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(1) #1234
  train_data = dataset.DATA(config.TRAIN_DIR) 
  train_sampler = torch.utils.data.RandomSampler(train_data) 
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=config.BATCH_SIZE,
      sampler=train_sampler,
      num_workers=4,
      drop_last=True,
      pin_memory=True)
  DEVICE = config.DEVICE

  netG = model.UNet_3Plus().float()
  netD = model.UNetDiscriminator().float()
  netG = netG.to(DEVICE)
  netD = netD.to(DEVICE)
  optD = torch.optim.Adam(netD.parameters(), lr=4e-5, betas=(0.5, 0.999)) 
  optG = torch.optim.Adam(netG.parameters(), lr=1e-5, betas=(0.5, 0.999)) #论文里这里是2e-5
  ## Trains
  losses = {
      'G_losses' : [],
      'D_losses' : [],
      'EPOCH_G_losses' : [],
      'EPOCH_D_losses' : [],
      'G_losses_eval' : []
  }

  netG, optG, netD, optD, epoch_checkpoint = utils.load_checkpoint("GAN",config.GAN_CHECKPOINT_DIR, netG, optG, netD, optD, DEVICE)
  netGAN = model.GAN(netG, netD)

  for epoch in range(epoch_checkpoint,config.NUM_EPOCHS+1):
    print('\n')
    print('#'*8,f'EPOCH-{epoch}','#'*8)
    losses['EPOCH_G_losses'] = []
    losses['EPOCH_D_losses'] = []
    engine.train(train_loader, netGAN, netD, optG, optD, device=DEVICE, losses=losses)
    if epoch%1==0:
      utils.create_checkpoint("GAN",epoch, netG, optG, netD, optD, max_checkpoint=config.KEEP_CKPT, save_path = config.GAN_CHECKPOINT_DIR)
    utils.plot_some("GAN",train_data, netG, DEVICE, epoch)
    gc.collect()
  
if __name__=='__main__':
  map_fn()