'''
    Automatic bias removal for Face Recognition
    Author: Omar U. Florez
    November, 2018
''' 


import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from convae import AutoEncoder
from genderpred import GenderPredictor
from vggface import VGGface
from dataset_loader import get_loader
import os
import time
import datetime

import ipdb

import argparse
parser = argparse.ArgumentParser()
parser.prog = 'main_train'
parser.add_argument('-gpu', type=str)
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_folder     = '/Users/ost437/Documents/OneDrive/workspace/datasets/celebrity/'
celeba_imgpath  = data_folder + 'data/images-dpmcrop-train/'
train_labelfile = data_folder + 'data/list_attr_celeba.txt'
proto_samepath  = data_folder + 'data/images-sameproto-aligned/' # aligned prototypes (same gender)
proto_oppopath  = data_folder + 'data/images-opGproto-aligned/'  # aligned prototypes (opposite gender)

use_cuda        = False
device          = 'cpu' if not use_cuda else 'gpu'
learning_rate   = 0.0002
num_epochs      = 10
batch_size      = 8

data_loader = get_loader(image_path=celeba_imgpath,
                         proto_same_path=proto_samepath,
                         proto_oppo_path=proto_oppopath,
                         metadata_path=train_labelfile,
                         image_size=(224, 224),
                         batch_size=batch_size,
                         num_workers=1)

#-----------------------------------------------------------------------------------------------------------------------
## autoencoder:
cae = AutoEncoder()
# to reload partially trained model:
# cae.load_state_dict(torch.load(os.path.join('../model/conv-autoencoder--{}.pkl'.format()))
if use_cuda:
    cae.cuda()

#-----------------------------------------------------------------------------------------------------------------------
## gender predictor
gpred = GenderPredictor()
gpred.load_state_dict(torch.load('../model/aux-gpred.pkl', map_location=device))
## freezing the gpred model
for param in gpred.parameters():
    param.requires_grad = False
gpred.eval()
if use_cuda:
    gpred.cuda()

#-----------------------------------------------------------------------------------------------------------------------
## VGG face rep:
vgg = VGGface()
vgg.load_state_dict(torch.load('../model/vggface.pt-adj-255.pkl', map_location=device))
## freezing the vgg model
for param in vgg.parameters():
    param.requires_grad = False
vgg.eval()
if use_cuda:
    vgg.cuda()


start_time = time.time()
optimizer = torch.optim.Adam(cae.parameters(), lr=learning_rate)

for epoch in range(0, 20):
    for i, (batch_x, batch_smG, batch_opG, batch_y) in enumerate(data_loader):
        if use_cuda:
            x_var = Variable(batch_x).cuda()
            sm_var = Variable(batch_smG).cuda()
            op_var = Variable(batch_opG).cuda()
        else:
            x_var = Variable(batch_x)
            sm_var = Variable(batch_smG)
            op_var = Variable(batch_opG)

        optimizer.zero_grad()

        # Models:
        #   x_var, sm_var, op_var:  torch.Size([8, 1, 224, 224])
        #   rec_sm, rec_op:         torch.Size([8, 1, 224, 224])
        #   gpred_sm, gpred_op:     torch.Size([8, 2])
        rec_sm, rec_op  = cae(x_var, sm_var, op_var)    #face autoencoder
        gpred_sm        = gpred(rec_sm)                 #gender prediction
        gpred_op        = gpred(rec_op)                 #gender prediction

        # Loss functions:
        loss_gender_sm = F.cross_entropy(input=gpred_sm,                        #same gender loss
                                         target=Variable(batch_y[:, 0]))
        loss_gender_op = F.cross_entropy(input=gpred_op,                        #oposite gender loss
                                         target=Variable(1-batch_y[:, 0]))
        loss_rec = g_loss_rec = torch.mean(torch.abs(x_var - rec_sm))           #reconstruction loss

        if epoch < 5:
            ## the first 5 epochs without matching loss
            loss_match = loss_rec
        else:
            #rep_vgg_sm, rep_vgg_op: torch.Size([8, 2622])                      #embedding of same reconstruction
            rep_vgg_sm = vgg(rec_sm)
            rep_vgg_op = vgg(rec_op)                                            #embedding of opposite reconstruction

            rep_vgg_orig = vgg(x_var).data
            rep_vgg_mean = (rep_vgg_sm + rep_vgg_op)/2.0

            loss_vgg_match = F.mse_loss(input=rep_vgg_mean,
                                        target=Variable(rep_vgg_orig),          #embedding of original input: x_var
                                        size_average=False)

            loss_match = loss_rec + 8.0*loss_vgg_match

        loss = loss_gender_sm + loss_gender_op + loss_match
        #ipdb.set_trace()

        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            elapsed = np.ceil(time.time() - start_time)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print ('Epoch [{}/{}], Iter [{}/{}] Elapsed [{}]  '
                   'Loss: {:.3f} {:.3f} {:.3f}  {:.3f} {:.3f}'.format(
                epoch+1, num_epochs, i+1, len(data_loader), elapsed,
                loss_gender_sm.data[0], loss_gender_op.data[0],
                loss_rec.data[0], loss_match.data[0],
                loss.data[0]))
            #break
        #break

torch.save(cae.state_dict(), '../conv-autoencoder-e20.pkl')
