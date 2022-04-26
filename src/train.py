import model
import config
import dataset
import utils
import engine
import torchvision.transforms as transforms
import torch
import warnings
warnings.filterwarnings('ignore')
import gc


def map_fn(index=None, flags=None):
  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(1234) #1234
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
  netDimg = model.ImgDiscriminator().float()
  netDgrad = model.GradDiscriminator().float()
  netG = netG.to(DEVICE)
  netDimg = netDimg.to(DEVICE)
  netDgrad = netDgrad.to(DEVICE)
  optDimg = torch.optim.Adam(netDimg.parameters(), lr=2e-5, betas=(0.9, 0.999))
  optDgrad = torch.optim.Adam(netDgrad.parameters(), lr=2e-5, betas=(0.9, 0.999)) 
  optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.9, 0.999))
  ## Trains
  losses = {
      'G_losses' : [],
      'Dimg_losses' : [],
      'EPOCH_G_losses' : [],
      'EPOCH_Dimg_losses' : [],
      'G_losses_eval' : []
  }

  netG, optG, netDimg, optDimg, netDgrad, optDgrad,epoch_checkpoint = utils.load_checkpoint("GAN",config.GAN_CHECKPOINT_DIR, netG, optG, netDimg, optDimg,netDgrad,optDgrad, DEVICE)
  # netG,epoch_checkpoint = utils.load_pretrained(config.GAN_CHECKPOINT_DIR, netG, DEVICE)
  netGAN = model.GAN(netG, netDimg, netDgrad)

  for epoch in range(epoch_checkpoint,config.NUM_EPOCHS+1):
    print('\n')
    print('#'*8,f'EPOCH-{epoch}','#'*8)
    losses['EPOCH_G_losses'] = []
    losses['EPOCH_D_losses'] = []
    engine.train(train_loader, netGAN, netDimg, netDgrad, optG, optDimg, optDgrad, device=DEVICE, losses=losses)
    if epoch%5==0:
      utils.create_checkpoint("GAN",epoch, netG, optG, netDimg, optDimg, netDgrad, optDgrad, max_checkpoint=config.KEEP_CKPT, save_path = config.GAN_CHECKPOINT_DIR)
    utils.plot_some("GAN",train_data, netG, DEVICE, epoch)
    gc.collect()
  
if __name__=='__main__':
  map_fn()