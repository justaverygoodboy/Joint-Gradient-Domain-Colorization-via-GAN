import model
import config
import dataset
import utils
import torch.nn as nn
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.FloatTensor') #这里加了.cuda
torch.manual_seed(1) #1234

train_data = dataset.DATA(config.TRAIN_DIR) 
train_sampler = torch.utils.data.RandomSampler(train_data) # 这里应该是随机打乱数据

train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=config.BATCH_SIZE,
      sampler=train_sampler,
      num_workers=6,
      drop_last=True,
      pin_memory=False)
DEVICE = config.DEVICE

net = model.UNet_3Plus_AE().float().to(DEVICE)

opt = torch.optim.Adam(net.parameters(), lr=2e-5, betas=(0.5, 0.999))

losses = {
      'MSE' : [],
  }

net, opt, epoch_checkpoint = utils.load_checkpoint_AE(config.CHECKPOINT_DIR, net, opt, DEVICE)

def train():
  batch = 0
  ## 迭代读取数据，直接分离了L和AB通道
  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  

      trainL = torch.tensor(trainL, device=DEVICE).float()
      trainAB = torch.tensor(trainAB, device=DEVICE).float()
      realLAB = torch.cat([trainL, trainAB], dim=1) 
      opt.zero_grad()
      predLAB = net(realLAB) 

      Loss_MSE = nn.MSELoss()(predLAB, realLAB) 
      Loss_MSE.backward()
      opt.step() 
      losses['MSE'].append(Loss_MSE.item())
      # Output training stats
      if batch % 10 == 0: 
        print('MSE: %.8f '% (Loss_MSE.item()))
def fn():
    for epoch in range(epoch_checkpoint,config.NUM_EPOCHS+1):
        print('\n')
        print('#'*8,f'EPOCH-{epoch}','#'*8)
        losses['MSE'] = []
        train()
        utils.create_checkpoint_AE(epoch, net, opt, max_checkpoint=config.KEEP_CKPT, save_path = config.CHECKPOINT_DIR)
        utils.plot_some_AE(train_data, net, DEVICE, epoch)

if __name__ == '__main__':
    fn()