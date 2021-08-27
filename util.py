import os
import torch
import torch.nn as nn
from torch import float32
from model import *

# Training parameters
useSave = False
useIdLoss = True
version = 15
lr = 2e-4
batch_size = 4
epochs = 100
wgt_cyc = 10.0
wgt_id = 5.0
direction = 'A2B'       # 'A2B' or 'B2A'

# ver4 : batch 2번마다 D가 1번 update되도록 변경
# ver5 : batch 10번마다 D가 1번 update되도록 변경
# ver6 : weight initialization 추가
# ver7 : wgt_cyc = 10, ID loss 사용하지 않음
# ver8 : ID loss 사용함
# ver9 : wgt_cyc = 20
# ver10 : G net 마지막을 Sigmoid -> Tanh로 변경, G architecture 일부 변경
# ver11 : D 마지막의 sigmoid 제거, GAN loss function을 BCE에서 MSE로 변경, batch 5번마다 D가 1번 update
# ver12 : D loss를 2로 나누어줌
# ver13 : wgt_cyc = 10, wgt_id = 5, batch 10번마다 D가 1번 update
# ver14 : batch 5번마다 D가 1번 update, epoch 3 이후부터는 D loss를 4로 나누어줌
# ver15 : batch 10번마다 D가 1번 update, D loss를 4로 나누어줌

# Directories
def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

dir_data = 'E:\horse2zebra'
dir_train = os.path.join(dir_data, 'train')
dir_test = os.path.join(dir_data, 'test')

ckpt_dir = 'C:/Users/msKim/Desktop/cyclegan/ckpt'
ckpt_use_dir = 'C:/Users/msKim/Desktop/cyclegan/ckptuse'
log_dir = 'C:/Users/msKim/Desktop/cyclegan/log'
result_dir = 'C:\\Users\\msKim\\Desktop\\cyclegan\\result'
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

# Helper Functions
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# segmentation data를 output image의 크기에 맞게 crop
def cropimg(seg, output):
    s1, s2 = seg.size()[-2:]
    o1, o2 = output.size()[-2:]
    i1, i2 = (s1-o1)//2, (s2-o2)//2
    segcrop = seg[:, :, i1 : (i1+o1), i2 : (i2+o2)]
    return segcrop

# 네트워크 저장
def save(ckpt_dir, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch, ver):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                "%s/ver%d_model_epoch%d.pth" % (ckpt_dir, ver, epoch))

def saveG(ckpt_dir, netG, optimG, epoch, ver):
    torch.save({'netG': netG.state_dict(), 'optimG': optimG.state_dict()},
                "%s/ver%d_modelG_epoch%d.pth" % (ckpt_dir, ver, epoch))

# 네트워크 불러오기
def load(ckpt_dir, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    print("USED MODEL :  %s" % ckpt_lst[-1])
    netG_a2b.load_state_dict(dict_model['netG_a2b'])
    netG_b2a.load_state_dict(dict_model['netG_b2a'])
    optimG.load_state_dict(dict_model['optimG'])

    netD_a.load_state_dict(dict_model['netD_a'])
    netD_b.load_state_dict(dict_model['netD_b'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch

# output pixels의 값을 0또는 1으로 변환
def makePredict(output):
    result = torch.zeros_like(output)
    result[output > 0.4] = 1
    return result

def F1_score(output, seg):
    tp = (seg * output).sum().to(torch.float32)
    tn = ((1 - seg) * (1 - output)).sum().to(torch.float32)
    fp = ((1 - seg) * output).sum().to(torch.float32)
    fn = (seg * (1 - output)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return float(f1)

def Dice_score(output, seg, mode='kid'):
    seg0 = torch.zeros_like(seg)
    seg1 = torch.zeros_like(seg)
    seg2 = torch.zeros_like(seg)
    seg0[seg == 0] = 1
    seg1[seg == 1] = 1
    seg2[seg == 2] = 1
    score0 = F1_score(output[:, 0, :, :], seg0)
    score1 = F1_score(output[:, 1, :, :], seg1)
    if (mode == 'all'):
        score2 = F1_score(output[:, 2, :, :], seg2)
        score = (score0 + score1 + score2) / 3.0
    else :
        score = (score0 + score1) / 2.0

    return float(score)