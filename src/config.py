import os
import torch
DATA_DIR = '../input/'
OUT_DIR = '../result/'
MODEL_DIR = '../models/'
GAN_CHECKPOINT_DIR = '../checkpoint/'
AE_CHECKPOINT_DIR = '../ae_checkpoint/'

AE_TRAIN_DIR = DATA_DIR+"toy/"
TRAIN_DIR = DATA_DIR+"train_cifar10/"  # UPDATE
TEST_DIR = DATA_DIR+"test_cifar10/" # UPDATE

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GAN_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# DATA INFORMATION
IMAGE_SIZE = 32 #初始是224
BATCH_SIZE = 50 #初始是64，不过由于后面有些汽车的没学到导致表现不太好，改50试试
GRADIENT_PENALTY_WEIGHT = 10
NUM_EPOCHS = 1000#初始是10
KEEP_CKPT = 2 #init 2
# save_model_path = MODEL_DIR

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  assert False
  DEVICE = 'cpu'

