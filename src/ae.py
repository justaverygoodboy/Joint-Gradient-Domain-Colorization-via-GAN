import model
import config
import dataset
import utils
import torch.nn as nn
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def train(train_loader,net,opt,losses,DEVICE):
  batch = 0
  ## 迭代读取数据，直接分离了L和AB通道
  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  
      L_Z = torch.randn(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE).to(DEVICE)
      L_AB = torch.randn(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE).to(DEVICE)
      trainL = torch.tensor(trainL, device=DEVICE).float()
      trainAB = torch.tensor(trainAB, device=DEVICE).float()
      N_L = trainL + 0.1*L_Z
      N_AB = trainAB + 0.1*L_AB
      realLAB = torch.cat([trainL, trainAB], dim=1)
      noiseLAB = torch.cat([N_L,N_AB],dim=1)
      # noiseLAB = torch.cat([trainL ,L_AB],dim=1) #灰图加噪声版本
      opt.zero_grad()
      predLAB = net(noiseLAB)
      Loss = nn.MSELoss()(predLAB,realLAB)
      Loss.backward()
      opt.step() 
      losses['Loss'].append(Loss.item())
      if batch % 10 == 0: 
        print('L: %.8f'% (Loss.item()))

def fn():
    torch.set_default_tensor_type('torch.FloatTensor') #这里加了.cuda
    torch.manual_seed(1) #1234
    torch.set_num_threads(1) 
    train_data = dataset.DATA(config.TRAIN_DIR) 
    train_sampler = torch.utils.data.RandomSampler(train_data) # 这里应该是随机打乱数据
    train_loader = torch.utils.data.DataLoader(
          train_data,
          batch_size=config.BATCH_SIZE,
          sampler=train_sampler,
          num_workers=1,
          drop_last=True,
          pin_memory=True)
    DEVICE = config.DEVICE
    net = model.UNet_3Plus_AE().float().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.5, 0.999))
    losses = {
          'Loss' : [],
      }

    # net, opt, epoch_checkpoint = utils.load_checkpoint("AE",config.AE_CHECKPOINT_DIR, net, opt,None,None, DEVICE)
    for epoch in range(1,config.NUM_EPOCHS+1):
        print('\n')
        print('#'*8,f'EPOCH-{epoch}','#'*8)
        losses['Loss'] = []
        train(train_loader,net,opt,losses,DEVICE)
        if (epoch % 10 == 0):
          utils.create_checkpoint("AE",epoch, net, opt, None, None, max_checkpoint=config.KEEP_CKPT, save_path = config.AE_CHECKPOINT_DIR)
        utils.plot_some("AE",train_data, net, DEVICE, epoch)

if __name__ == '__main__':
    fn()