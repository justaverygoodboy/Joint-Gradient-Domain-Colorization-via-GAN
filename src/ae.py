import model
import config
import dataset
import utils
import torch.nn as nn
import torch
from tqdm import tqdm
from lossfunc import GradientLoss
import warnings
warnings.filterwarnings('ignore')

def train(train_loader,net,opt,losses,DEVICE):
  batch = 0
  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  
      Z = torch.randn(64,2,config.IMAGE_SIZE,config.IMAGE_SIZE).to(DEVICE)
      trainL = torch.tensor(trainL, device=DEVICE).float()
      trainAB = torch.tensor(trainAB, device=DEVICE).float()
      realLAB = torch.cat([trainL, Z], dim=1)
      # noiseLAB = realLAB+0.3*Z

      opt.zero_grad()

      predAB = net(realLAB)
      # Loss_Gradient = GradientLoss()(predLAB,realLAB)
      Loss_L2 = nn.MSELoss()(predAB,trainAB)
      Loss = Loss_L2
      Loss.backward()
      opt.step() 
      losses['Loss'].append(Loss.item())
      if batch % 10 == 0: 
        print('L: %.8f'% (Loss.item()))

def fn():
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(1234) #1234
    torch.set_num_threads(1) 
    train_data = dataset.DATA(config.AE_TRAIN_DIR) 
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_loader = torch.utils.data.DataLoader(
          train_data,
          batch_size=64,
          sampler=train_sampler,
          num_workers=4,
          drop_last=True,
          pin_memory=True)
    DEVICE = config.DEVICE
    net = model.DAE().float().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    losses = {
          'Loss' : [],
      }
    net, epoch_checkpoint = utils.load_checkpoint("AE",config.AE_CHECKPOINT_DIR, net, opt,None,None,None,None, DEVICE)
    for epoch in range(1,config.NUM_EPOCHS+1):
        print('\n')
        print('#'*8,f'EPOCH-{epoch}','#'*8)
        losses['Loss'] = []
        # utils.plot_some("AE",train_data, net, DEVICE, epoch)
        train(train_loader,net,opt,losses,DEVICE)
        if (epoch % 10 == 0):
          utils.create_checkpoint("AE",epoch, net, opt, None,None,None, None, max_checkpoint=config.KEEP_CKPT, save_path = config.AE_CHECKPOINT_DIR)
        utils.plot_some("AE",train_data, net, DEVICE, epoch)

if __name__ == '__main__':
    fn()