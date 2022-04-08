import model
import config
import dataset
import utils
import torch
import warnings
warnings.filterwarnings('ignore')

def test():
  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(1234) #1234
  test_data = dataset.DATA(config.TEST_DIR) 
  test_sampler = torch.utils.data.RandomSampler(test_data) 
  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=config.BATCH_SIZE,
      sampler=test_sampler,
      num_workers=4,
      drop_last=True,
      pin_memory=True)
  DEVICE = config.DEVICE

  netG = model.UNet_3Plus().float()
  netG = netG.to(DEVICE)

  netG, epoch_checkpoint = utils.load_checkpoint("test",config.GAN_CHECKPOINT_DIR, netG, None, None, None, DEVICE)
  utils.plot_some("test",test_data, netG, DEVICE,0)
    
  
if __name__=='__main__':
  test()