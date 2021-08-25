import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import itertools
from torch.utils.tensorboard import SummaryWriter
from util import *
from dataloader import *

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
#writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))

gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(300)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Dataset & Network
transform_train = transforms.Compose([ToTensor()])
transform_inv = transforms.Compose([ToNumpy()])
dataset_train = cyclegan_Dataset(dir_train, transform=transform_train, direction=direction)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)

"""
# Create Test set
transform_test = transforms.Compose([Normalization(), ToTensor()])
dataset_test = cyclegan_Dataset(dir_test, transform=transform_test, direction=direction)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)
"""

G_a2b = Generator(in_chs=3, out_chs=3, num_blk=9).to(device)
G_b2a = Generator(in_chs=3, out_chs=3, num_blk=9).to(device)
D_a = Discriminator(in_chs=3, out_chs=1).to(device)
D_b = Discriminator(in_chs=3, out_chs=1).to(device)

# weight initialization ???
init_weights(G_a2b)
init_weights(G_b2a)
init_weights(D_a)
init_weights(D_b)

# Loss Functions & Optimizer
fn_GAN = nn.MSELoss().to(device)
fn_cyc = nn.L1Loss().to(device)
fn_id = nn.L1Loss().to(device)

optimG = torch.optim.Adam(itertools.chain(G_a2b.parameters(), G_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(itertools.chain(D_a.parameters(), D_b.parameters()), lr=lr, betas=(0.5, 0.999))

# 학습 전 저장된 네트워크가 있다면 불러오기
st_epoch = 0
if (useSave):
    G_a2b, G_b2a, D_a, D_b, optim_G_a2b, optim_G_b2a, optim_D_a, optim_D_b, st_epoch = \
        load(ckpt_dir=ckpt_use_dir, netG_a2b=G_a2b, netG_b2a=G_b2a, netD_a=D_a, netD_b=D_b,
            optimG=optimG, optimD=optimD)

# Training
print("TRAINING STARTS")
for epoch in range(st_epoch, epochs):
    G_a2b.train()
    G_b2a.train()
    D_a.train()
    D_b.train()
    loss_G_a2b_GAN_train = []
    loss_G_b2a_GAN_train = []
    loss_G_a_cyc_train = []
    loss_G_b_cyc_train = []
    loss_G_a_id_train = []
    loss_G_b_id_train = []
    loss_D_a_train = []
    loss_D_b_train = []

    for batch, data in enumerate(loader_train, 1):
        # forward pass
        in_A = data['A'].to(device)
        in_B = data['B'].to(device)
        out_B = G_a2b(in_A)
        out_A = G_b2a(in_B)
        cyc_A = G_b2a(out_B)
        cyc_B = G_a2b(out_A)

        # backward D
        if (batch % 5 == 1):
            optimD.zero_grad()

            pred_real_a = D_a(in_A)
            pred_fake_a = D_a(out_A.detach())
            loss_D_a_real = fn_GAN(pred_real_a, torch.ones_like(pred_real_a))
            loss_D_a_fake = fn_GAN(pred_fake_a, torch.zeros_like(pred_fake_a))
            loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

            pred_real_b = D_b(in_B)
            pred_fake_b = D_b(out_B.detach())
            loss_D_b_real = fn_GAN(pred_real_b, torch.ones_like(pred_real_b))
            loss_D_b_fake = fn_GAN(pred_fake_b, torch.zeros_like(pred_fake_b))
            loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

            loss_D = loss_D_a + loss_D_b
            loss_D.backward()
            optimD.step()

        # backward G
        optimG.zero_grad()
        
        pred_fake_a = D_a(out_A)
        pred_fake_b = D_b(out_B)

        if (useIdLoss):
            id_A = G_b2a(in_A)
            id_B = G_a2b(in_B)
            loss_G_a_id = fn_id(in_A, id_A)
            loss_G_b_id = fn_id(in_B, id_B)
        else :
            loss_G_a_id = 0
            loss_G_b_id = 0

        loss_G_a2b_GAN = fn_GAN(pred_fake_b, torch.ones_like(pred_fake_b))
        loss_G_b2a_GAN = fn_GAN(pred_fake_a, torch.ones_like(pred_fake_a))
        loss_G_a_cyc = fn_cyc(in_A, cyc_A)
        loss_G_b_cyc = fn_cyc(in_B, cyc_B)

        loss_G = loss_G_a2b_GAN + loss_G_b2a_GAN + \
                 (wgt_cyc * (loss_G_a_cyc + loss_G_b_cyc)) + \
                 (wgt_id * (loss_G_a_id + loss_G_b_id))

        loss_G.backward()
        optimG.step()

        # loss 계산
        loss_G_a2b_GAN_train += [loss_G_a2b_GAN.item()]
        loss_G_b2a_GAN_train += [loss_G_b2a_GAN.item()]
        loss_G_a_cyc_train += [loss_G_a_cyc.item()]
        loss_G_b_cyc_train += [loss_G_b_cyc.item()]
        if (useIdLoss):
            loss_G_a_id_train += [loss_G_a_id.item()]
            loss_G_b_id_train += [loss_G_b_id.item()]
        loss_D_a_train += [loss_D_a.item()]
        loss_D_b_train += [loss_D_b.item()]

        if (batch % 10 == 0) :
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                  "G CYC : %.4f, %.4f | G GAN : %.4f, %.4f | "
                  "D : %.4f, %.4f " %
                  (epoch + 1, epochs, batch, num_batch_train,
                   loss_G_a_cyc.item(), loss_G_b_cyc.item(),
                   loss_G_a2b_GAN.item(), loss_G_b2a_GAN.item(),
                   loss_D_a.item(), loss_D_b.item()))
        
        if (batch % 20 == 0) :
            in_A = transform_inv(in_A)
            in_B = transform_inv(in_B)
            out_A = transform_inv(out_A)
            out_B = transform_inv(out_B)

            batchnum = num_batch_train * epoch + batch
            res_path = os.path.join(result_dir_train, 'ver_%03d' % version)
            makeDir(res_path)
            plt.imsave(os.path.join(res_path, 'epoch%03d_batch%03d_inA.png' % (epoch+1, batch)), in_A[0])
            plt.imsave(os.path.join(res_path, 'epoch%03d_batch%03d_inB.png' % (epoch+1, batch)), in_B[0])
            plt.imsave(os.path.join(res_path, 'epoch%03d_batch%03d_outA.png' % (epoch+1, batch)), out_A[0])
            plt.imsave(os.path.join(res_path, 'epoch%03d_batch%03d_outB.png' % (epoch+1, batch)), out_B[0])

            writer_train.add_image('in_A', in_A, batchnum, dataformats='NHWC')
            writer_train.add_image('in_B', in_B, batchnum, dataformats='NHWC')
            writer_train.add_image('out_A', out_A, batchnum, dataformats='NHWC')
            writer_train.add_image('out_B', out_B, batchnum, dataformats='NHWC')
    
    # epoch 1번 돈 후 loss 기록
    writer_train.add_scalar('loss_G_a_cyc', np.mean(loss_G_a_cyc_train), epoch+1)
    writer_train.add_scalar('loss_G_b_cyc', np.mean(loss_G_b_cyc_train), epoch+1)
    writer_train.add_scalar('loss_G_a2b_GAN', np.mean(loss_G_a2b_GAN_train), epoch+1)
    writer_train.add_scalar('loss_G_b2a_GAN', np.mean(loss_G_b2a_GAN_train), epoch+1)
    writer_train.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch+1)
    writer_train.add_scalar('loss_D_b', np.mean(loss_D_b_train), epoch+1)

    if (True):
        save(ckpt_dir=ckpt_dir, netG_a2b=G_a2b, netG_b2a=G_b2a, netD_a=D_a, netD_b=D_b, optimG=optimG, optimD=optimD, epoch=epoch, ver=version)
    
writer_train.close()