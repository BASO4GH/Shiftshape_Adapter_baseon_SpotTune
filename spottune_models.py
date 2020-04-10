#Merged with adapter #Spottune_models.py
#tensorflow




import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

from torch.nn import init

class Linear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # self.init_std = Parameter(torch.rand(1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input_a = list(input.size())[0]
        input_b = list(input.size())[1]
        w1 = torch.empty(input_b,int(self.in_features))
        # nn.init.normal_(w1, mean=0, std=self.init_std.data[0])
        nn.init.eye_(w1)
        # w1 = w1.float()
        
        input = torch.matmul(input.cpu(), w1)
        input = F.linear(input.cuda(), self.weight, self.bias)

        w2 = torch.empty(int(self.out_features),input_b)
        # nn.init.normal_(w2, mean=0, std=self.init_std.data[0])
        nn.init.eye_(w2)
        # w2 = w2.float()
        input = torch.matmul(input.cpu(), w2)

        input = input.cuda()
        return input

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# class FeedforwardAdapter(nn.Module):
#     def __init__(self, hid_siz_per=0.5, init_scale=1e-3):
#         super(FeedforwardAdapter, self).__init__()
#         self.hid_siz_per = hid_siz_per
#         self.init_scale = init_scale
        
#     def forward(self, input_tensor):
#         in_size0 = list(input_tensor.size())[0]
#         in_size1 = list(input_tensor.size())[1]
#         in_size_e = list(input_tensor.size())[-1]
        
#         hidden_size = int(round(in_size_e*self.hid_siz_per))
#         if hidden_size<=1:
#           hidden_size = 1
#         # print(self.hid_siz_per)
#         # w1 = torch.empty(in_size0,in_size1,in_size_e,hidden_size)
#         # nn.init.normal_(w1, mean=0, std=self.init_scale)
#         w1 = torch.empty(in_size_e,hidden_size)
#         # print('w1_shape: '+str(w1.size()))
#         # print('input_tensor_shape: '+str(input_tensor.size()))
#         nn.init.eye_(w1)
#         w1 = w1.float()
#         # b1 = torch.empty(in_size0,in_size1,1,hidden_size)
#         # nn.init.zeros_(b1)
#         net = torch.matmul(input_tensor.cpu(), w1)
#         # net = gelu(net)
#         net = F.relu(net)
#         net = torch.tensor(net, requires_grad=True)

#         # w2 = torch.empty(in_size0,in_size1,hidden_size,in_size_e)
#         # nn.init.normal_(w2, mean=0, std=self.init_scale)
#         w2 = torch.empty(hidden_size,in_size_e)
#         nn.init.eye_(w2)
#         w2 = w2.float()
#         # b2 = torch.empty(in_size0,in_size1,1,in_size_e)
#         # nn.init.zeros_(b2)
#         net = torch.matmul(net, w2)

#         res = net+input_tensor.cpu()
#         res = res.cuda()
#         return res


class FeedforwardAdapter(nn.Module):
    def __init__(self, hid_siz_per=0.5, adjSize = (10,10), init_scale=1e-3):
        super(FeedforwardAdapter, self).__init__()
        self.hid_siz_per = hid_siz_per
        self.init_scale = init_scale
        self.adjustSize = adjSize
        w = torch.empty(self.adjustSize)
        nn.init.normal_(w, mean=0, std=self.init_scale)
        # w = torch.Tensor(w).cuda()
        self.adjustTensor = w
        self.adjustTensor = self.adjustTensor.cuda()
        
    def forward(self, input_tensor):
        in_size0 = list(input_tensor.size())[0]
        in_size1 = list(input_tensor.size())[1]
        in_size_e = list(input_tensor.size())[-1]
        
        hidden_size = int(round(in_size_e*self.hid_siz_per))
        if hidden_size<=1:
          hidden_size = 1
        w1_sa = torch.empty(in_size_e, self.adjustSize[0])
        w1_bh = torch.empty(self.adjustSize[1], hidden_size)
        nn.init.eye_(w1_sa)
        nn.init.eye_(w1_bh)
        w1 = torch.matmul(torch.matmul(w1_sa,self.adjustTensor.cpu()), w1_bh)
        w1 = w1.float()

        net = torch.matmul(input_tensor.cpu(), w1)
        # net = gelu(net)
        net = F.relu(net)

        w2 = torch.empty(hidden_size,in_size_e)
        nn.init.normal_(w2, mean=0, std=self.init_scale)
        w2 = w2.float()

        net = torch.matmul(net, w2)

        res = net+input_tensor.cpu()
        res = res.cuda()
        return res



class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual*0),1)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1
    

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(int(planes))

        self.adapter = nn.Sequential(FeedforwardAdapter())  #adapter#ongoingversion

        self.conv2 = nn.Sequential(nn.ReLU(True), conv3x3(planes, planes))
        self.bn2 = nn.BatchNorm2d(int(planes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = F.relu(out)   #original
        out = self.adapter(out)  #adapter#ongoingversion

        out = self.conv2(out)
        y = self.bn2(out)

        return y

class ResNet(nn.Module):
    def __init__(self, block, layers, num_class = 10):
        super(ResNet, self).__init__()

        factor = 1
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor))
        self.bn1 = nn.BatchNorm2d(int(32*factor))
        self.relu = nn.ReLU(inplace=True)

        strides = [2, 2, 2]
        filt_sizes = [64, 128, 256]
        self.blocks, self.ds = [], []
        self.parallel_blocks, self.parallel_ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
    
        self.in_planes = int(32*factor)
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.parallel_blocks.append(nn.ModuleList(blocks))
            self.parallel_ds.append(ds)
        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)
        self.parallel_ds = nn.ModuleList(self.parallel_ds)

        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256*factor)), nn.ReLU(True))   #original
        # self.bn2 = nn.BatchNorm2d(int(256*factor))  #adapter

        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        # self.adapter = nn.Sequential(FeedforwardAdapter()) #adapter
        self.linear = nn.Linear(int(256*factor), num_class)   #original
        # self.linear = Linear(int(256*factor*0.9), int(num_class*0.95))   #adapter
        self.layer_config = layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.bn1(self.conv1(x))   #original
        # x = self.conv1(x)         #adapter
        # x = feedforward_adapter(x)     #adapter
        # x = self.bn1(x)           #adapter
        return x                 

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = DownsampleB(self.in_planes, planes * block.expansion, 2)

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return layers, downsample

    def forward(self, x, policy=None):
        t = 0
        x = self.seed(x)      #original
        # x = self.conv1(x)         #adapter
        # x = feedforward_adapter(x)     #adapter
        #policy=None #Test we use without SpotTune Policy
        if policy is not None:
            for segment, num_blocks in enumerate(self.layer_config):
                    # x = feedforward_adapter(x)  #adapter
                    for b in range(num_blocks):
                        action = policy[:,t].contiguous()
                        action_mask = action.float().view(-1,1,1,1)

                        residual = self.ds[segment](x) if b==0 else x   #original
                        output = self.blocks[segment][b](x)
			
                        residual_ = self.parallel_ds[segment](x) if b==0 else x   #original
                        output_ = self.parallel_blocks[segment][b](x)
                        # print('f1: '+str((residual + output).size()))
                        # print('f2: '+str((residual_ + output_).size()))
                        f1 = F.relu(residual + output)    #original
                        f2 = F.relu(residual_ + output_)    #original
                        # f1 = feedforward_adapter(residual + output) #adapter
                        # f2 = feedforward_adapter(residual_ + output_) #adapter

                        x = f1*(1-action_mask) + f2*action_mask
                        t += 1
                    # x = feedforward_adapter(x)  #adapter
            
        else:
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b==0 else x
                    output = self.blocks[segment][b](x)
                    x = F.relu(residual + output) #Original
                    # x = feedforward_adapter(residual + output)  #adapter
                    
                    t += 1

                # x = feedforward_adapter(x)  #adapter


        x = self.bn2(x)
        # x = self.adapter(x) #adapter
        # x = feedforward_adapter(x)    #adapter
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = feedforward_adapter_2d(x)
        x = self.linear(x)    #original

        return x

def resnet26(num_class=10, blocks=BasicBlock):
    return  ResNet(blocks, [4,4,4], num_class)
	


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + torch.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
  return x * cdf

#hidden_size = {1, 0.8, 0.6, 0.4, 0.2, 0.1}(1e-3)
def feedforward_adapter(input_tensor, hidden_size=64, init_scale=1e-3):
  in_size0 = list(input_tensor.size())[0]
  in_size1 = list(input_tensor.size())[1]
  in_size_e = list(input_tensor.size())[-1]
  # hidden_size = int(round(in_size_e*0.1))
  hidden_size = hidden_size
  #origin np.random.normal
  # w1 = Variable(torch.tensor(np.random.uniform(0,init_scale,[in_size0,in_size1,in_size_e,hidden_size])), requires_grad=True) 
  # b1 = Variable(torch.zeros([in_size0,in_size1,1,hidden_size]), requires_grad=True)
  # w1 = w1.float()
  
  w1 = torch.empty(in_size0,in_size1,in_size_e,hidden_size)
  # nn.init.normal_(w1, mean=0, std=init_scale)
  nn.init.eye_(w1)
  # nn.init.xavier_uniform_(w1, gain=1.0)
  w1 = w1.float()

  # b1 = Variable(torch.zeros([in_size0,in_size1,1,hidden_size]), requires_grad=True) #original
  b1 = torch.empty(in_size0,in_size1,1,hidden_size)
  nn.init.zeros_(b1)
  
  net = torch.matmul(input_tensor.cpu(), w1)+b1
  # net = gelu(net)
  net = F.relu(net)

  #origin np.random.normal
  # w2 = Variable(torch.tensor(np.random.uniform(0,init_scale,[in_size0,in_size1,hidden_size,in_size_e])), requires_grad=True)
  # w2 = w2.float()
  w2 = torch.empty(in_size0,in_size1,hidden_size,in_size_e)
  # nn.init.normal_(w2, mean=0, std=init_scale)
  nn.init.eye_(w2)
  # nn.init.xavier_uniform_(w2, gain=1.0)
  w2 = w2.float()

  # b2 = Variable(torch.zeros([in_size0,in_size1,1,in_size_e]), requires_grad=True) #origianl
  b2 = torch.empty(in_size0,in_size1,1,in_size_e)
  nn.init.zeros_(b2)
  net = torch.matmul(net, w2)+b2

  res = net+input_tensor.cpu()
  res = res.cuda()
  return res
  #ValueError: Dimensions must be equal, but are 64 and 32 for 'add' (op: 'AddV2') with input shapes: [128,64,64,32], [128,32,64,64].
  

def feedforward_adapter_2d(input_tensor, hidden_size=64, init_scale=1e-3):
  in_size0 = list(input_tensor.size())[0]
  in_size_e = list(input_tensor.size())[-1]
  # hidden_size = int(round(in_size_e*0.1))
  hidden_size = 1

  w1 = torch.empty(in_size_e,hidden_size)
  nn.init.normal_(w1, mean=0, std=init_scale)
  w1 = w1.float()
  b1 = torch.empty(1,hidden_size)
  nn.init.zeros_(b1)
  
  net = torch.matmul(input_tensor.cpu(), w1)+b1
  # net = gelu(net)
  net = F.relu(net)

  w2 = torch.empty(hidden_size,in_size_e)
  nn.init.normal_(w2, mean=0, std=init_scale)
  w2 = w2.float()
  b2 = torch.empty(1,in_size_e)
  nn.init.zeros_(b2)
  net = torch.matmul(net, w2)+b2

  res = net+input_tensor.cpu()
  res = res.cuda()
  return res


def get_adapter(function_string):
  if not function_string:
    return None

  fn = function_string.lower()
  if fn == "feedforward_adapter":
    return feedforward_adapter
  else:
    raise ValueError("Unsupported adapters: %s" % fn)