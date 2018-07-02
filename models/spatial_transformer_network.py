"""

spatial transformer and localizer for caltech birds,

constrain the transform parameters to perform cropping

the localization network output shape should be num_batch * 2* N,

where N is the number of transformers, all transformers share one localization network

"""
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math

class Localise(nn.Module):
    # according to https://arxiv.org/pdf/1506.02025.pdf,
    # add this layer before last pooling layer of convnet, 
    # add 3 weight layers 1x1 conv (128)-  fc(128) - fc(2N)

  def __init__(self, num_transformers, inplanes):
      super(Localise, self).__init__()
      self.N = num_transformers
      self.conv = nn.Conv2d(inplanes, 128, kernel_size=1, stride=1, padding=0, bias=False)
      self.bn_conv = nn.BatchNorm2d(128)
      self.fc_1 = nn.Linear(128*7*7, 128)
      self.bn_fc_1 = nn.BatchNorm1d(128)
      self.fc_2 = nn.Linear(128, 2*self.N)
  
  
      self.conv.weight.data.normal_(0, math.sqrt(2./(1*1* 128) ))
      self.bn_conv.weight.data.fill_(1)
      self.bn_conv.bias.data.zero_()
      nn.init.kaiming_normal(self.fc_1.weight)
      self.fc_1.bias.data.zero_()
      self.bn_fc_1.weight.data.fill_(1)
      self.bn_fc_1.bias.data.zero_()
      self.fc_2.weight.data.zero_()
      self.fc_2.bias.data.zero_()
  
  def g_theta(self, theta):
      minibatch_size = theta.size()[0]
      eye = torch.eye(2)* 0.7
      eye = eye.unsqueeze(0).repeat(minibatch_size, 1,1)
      theta = theta.view(-1, self.N, 2)
      thetas = []
      for i in range(self.N):
          translation_coeff = theta[:, i, :].unsqueeze(-1)
          thetas.append(torch.cat([eye.cuda(), translation_coeff], -1))
      return thetas

  def forward(self,x):
      #out minibatch x 2N
      x = self.conv(x)
      x = F.relu(self.bn_conv(x), inplace=True)
      x = x.view((x.size(0), -1))
      x = self.fc_1(x)
      x = F.relu(self.bn_fc_1(x), inplace=True)
      
      out = F.tanh(self.fc_2(x))* 0.3

      return self.g_theta(out)

        
  



class STN(nn.Module):
  def __init__(self,out_size):
        super(STN, self).__init__()
        self.out_size = out_size

  def forward(self, x, theta):

        return self.spatial_transform(x, theta, self.out_size)

  def spatial_transform(self, U, theta, out_size):
    #according to tensorflow version https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py

    def _crop_meshgrid(height, width):
        x_t = torch.linspace(-1,1 , width).repeat(height,1)
        y_t = torch.linspace(-1,1, height).view(-1,1).repeat(1, width)

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        #cropping wouldnt need the third coodinate 1
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def _interpolate(im, x, y, out_size):
        minibatch_size,channels, height, width = im.size()
        
        #rescale to 0 - height|weright
        x = (x+1.0)*(width-1) / 2.0
        y = (y+1.0)*(height-1) / 2.0

        max_y = height -1
        max_x = width - 1

        x0 = torch.floor(x).long()
        x1 = x0+1
        y0 = torch.floor(y).long()
        y1 = y0+1

        #clip
        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        row_base = width
        idx_a = (y0* row_base + x0).unsqueeze(1).repeat(1,channels,1)
        idx_b = (y1* row_base + x0).unsqueeze(1).repeat(1,channels,1)
        idx_c = (y0* row_base + x1).unsqueeze(1).repeat(1,channels,1)
        idx_d = (y1* row_base + x1).unsqueeze(1).repeat(1,channels,1)

        im_flat = im.view(minibatch_size,channels, -1)
        im_flat = im_flat.float()
        Ia = torch.gather(im_flat, 2, idx_a)
        Ib = torch.gather(im_flat, 2, idx_b)
        Ic = torch.gather(im_flat, 2, idx_c)
        Id = torch.gather(im_flat, 2, idx_d)

        x0_f = x0.float()
        x1_f = (x0+1).float()
        y0_f = y0.float()
        y1_f = (y0+1).float()

        wa = ((x1_f - x)*(y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x)*(y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f)*(y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f)*(y - y0_f)).unsqueeze(1)

        output = Ia*wa + Ib*wb + Ic*wc + Id*wd
        return output.view(-1, channels, out_size[0], out_size[1])

    def _transform(theta, input_img, out_size):
        out_height, out_width = out_size
        grid = _crop_meshgrid(out_height, out_width)
        minibatch_size = input_img.size()[0]
        grid = grid.view(-1)
        grid = grid.repeat(minibatch_size)
        grid = grid.view(minibatch_size, 3, -1)
        theta = theta.view(-1, 2,3)
        T_g = torch.bmm(theta, grid.cuda())

        T_g = T_g
        x_s = T_g[:, 0]
        y_s = T_g[:, 1]

        transformed = _interpolate(input_img, x_s,y_s, out_size)
        return transformed

    return _transform(theta, U, out_size)


def test():
    import sys
    sys.path.append('../')
    from my_folder import MyImageFolder
    from neupeak.utils import webcv2 as cv2
    minibatch = 8
    theta = torch.from_numpy(np.array(
        [[0.5, 0, 0],
         [0, 0.5, 0]]
        ))
    theta = theta.expand(minibatch, 2,3).float()
    img_size = 224
    data_path = '/unsullied/sharefs/chenyaolin/cylfile/data_2/train'
    train_transform = transforms.Compose(
              [transforms.Scale((448,448)),
                  transforms.ToTensor()
                  ]
              )
    data = MyImageFolder(data_path, train_transform, data_cached= True)
    imgs  = torch.utils.data.DataLoader(data, batch_size= minibatch, shuffle = False)
    stn =  STN((img_size, img_size)) 
    stn = torch.nn.DataParallel(stn, device_ids=list(range(8)))

    stn.cuda()
    for batch in imgs:
        transformed = stn(batch[0].cuda(), theta)
        o = batch[0]

        t = transformed
        for i in range(minibatch):
           # from IPython import embed
           # embed()
            cv2.imshow('o',np.transpose( o[i].cpu().data.numpy()*255, (1,2,0)))
            cv2.imshow('transformed', np.transpose(t[i].cpu().data.numpy()*255,(1,2,0)))
            cv2.waitKey(0)



if __name__ == '__main__':
    test()

