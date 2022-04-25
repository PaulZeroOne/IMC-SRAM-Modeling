'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import QuantizeConv2d, QuantizeLinear
import numpy as np
import pandas as pd


class BasicBlock_quant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, w_bits=1, a_bits=1):
        super(BasicBlock_quant, self).__init__()
        self.conv1 = QuantizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                w_bits=w_bits, a_bits=a_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizeConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False,
                w_bits=w_bits, a_bits=a_bits)
        self.bn2 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuantizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                    w_bits=w_bits, a_bits=a_bits),
                #nn.BatchNorm2d(self.expansion*planes),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.bn2(out)
        return out


class Bottleneck_quant(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_quant, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_quant(nn.Module):
    def forward_pre_hook(self,module,inputs):
      
      #print("Pre hook executed input shape: ",inputs[0].shape)
      
      inputs_df_pre_hook = inputs[0].cpu()
      inputs_df_quant_pre_hook = inputs_df_pre_hook.type(torch.ByteTensor)
      #print(inputs_df_quant_pre_hook.dtype)
      #print(inputs_df_quant_pre_hook)
      inputs_df_quant_pre_hook_r=inputs_df_quant_pre_hook[:,0]
      np.save('inputs_df_quant_pre_hook_r.npy', inputs_df_quant_pre_hook_r.numpy())
      inputs_df_quant_pre_hook_g=inputs_df_quant_pre_hook[:,1]
      np.save('inputs_df_quant_pre_hook_g.npy', inputs_df_quant_pre_hook_g.numpy())
      inputs_df_quant_pre_hook_b=inputs_df_quant_pre_hook[:,2]
      np.save('inputs_df_quant_pre_hook_b.npy', inputs_df_quant_pre_hook_b.numpy())
      #np.savetxt("inputs_df_quant_pre_hook.txt", inputs_df_quant_pre_hook.reshape((4,-1)), fmt="%s", header=str(inputs_df_quant_pre_hook.shape))
      #TODO: Dump the values in a file
            

    def forward_hook(self,module,inputs,outputs):
      #print("Hook executed input shape: ",type(inputs),"output shape: ",outputs.shape)
   

      #print("Hook executed input shape: ",inputs[0].shape,"output shape: ",outputs.shape)
      
      inputs_df = inputs[0].cpu()
      inputs_df_quant = inputs_df.type(torch.ByteTensor)
      #print(inputs_df_quant.dtype)
      #print(inputs_df_quant)
      inputs_df_quant_r=inputs_df_quant[:,0]
      np.save('inputs_df_quant_r.npy', inputs_df_quant_r.numpy())
      inputs_df_quant_g=inputs_df_quant[:,1]
      np.save('inputs_df_quant_g.npy', inputs_df_quant_g.numpy())
      inputs_df_quant_b=inputs_df_quant[:,2]
      np.save('inputs_df_quant_b.npy', inputs_df_quant_b.numpy())
      #np.savetxt("inputs_df_quant.txt", inputs_df_quant.reshape((4,-1)), fmt="%s", header=str(inputs_df_quant.shape))
      #TODO: Dump the values in a file
      
      outputs_df = outputs.cpu()
      outputs_df_quant = outputs_df.type(torch.ByteTensor)
      #print(outputs_df_quant.dtype)
      #print(outputs_df_quant)
      outputs_df_quant_r=outputs_df_quant[:,0]
      np.save('outputs_df_quant_r.npy', outputs_df_quant_r.numpy())
      outputs_df_quant_g=outputs_df_quant[:,1]
      np.save('outputs_df_quant_g.npy', outputs_df_quant_g.numpy())
      outputs_df_quant_b=outputs_df_quant[:,2]
      np.save('outputs_df_quant_b.npy', outputs_df_quant_b.numpy())
      #np.savetxt("outputs_df_quant.txt", outputs_df_quant.reshape((4,-1)), fmt="%s", header=str(outputs_df_quant.shape))
      #TODO: Dump the values in a file
      
      
      
      

    def __init__(self, block, num_blocks, num_classes=10, w_bits=1, a_bits=1):
        super(ResNet_quant, self).__init__()
        self.in_planes = 64

        if num_classes == 1000:
            self.conv1 = QuantizeConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                w_bits=w_bits, a_bits=a_bits)
            self.conv1.register_forward_pre_hook(self.forward_pre_hook)
            self.conv1.register_forward_hook(self.forward_hook)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = QuantizeConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False,
                    w_bits=w_bits, a_bits=a_bits)
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, 
                w_bits=w_bits, a_bits=a_bits)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                w_bits=w_bits, a_bits=a_bits)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                w_bits=w_bits, a_bits=a_bits)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                w_bits=w_bits, a_bits=a_bits)
        self.linear = QuantizeLinear(512*block.expansion, num_classes, w_bits=w_bits, a_bits=a_bits)
        self.bn2 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        if num_classes == 1000:
            self.regime = {
                0: {'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-4},
                30: {'lr': 1e-2},
                60: {'lr': 1e-3}
                }
            
        else:
            self.regime = {
                0: {'optimizer': 'Adam', 'lr': 5e-3, 'momentum': 0, 'weight_decay': 0},
                101: {'lr': 1e-3},
                142: {'lr': 5e-4},
                184: {'lr': 1e-4},
                220: {'lr': 1e-5}
        }





    def _make_layer(self, block, planes, num_blocks, stride, w_bits=1, a_bits=1):
        layers = []
        if num_blocks == 1:
            layers.append(block(self.in_planes, planes, stride, w_bits, a_bits))
        else:
            layers.append(block(self.in_planes, planes, stride, w_bits, a_bits))
            self.in_planes = planes * block.expansion
            for i in range(num_blocks-2):
                layers.append(block(self.in_planes, planes, 1, w_bits, a_bits))
                self.in_planes = planes * block.expansion
            layers.append(block(self.in_planes, planes, 1, w_bits, a_bits))
            


        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.num_classes == 1000:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.num_classes == 1000:
            out = F.avg_pool2d(out, 7)
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.bn2(out)
        out = self.logsoftmax(out)
        return out


def ResNet18_quant(w_bits=1, a_bits=1, num_classes=10):
    return ResNet_quant(BasicBlock_quant, [2,2,2,2], w_bits=w_bits, a_bits=a_bits, num_classes=num_classes)

def ResNet34_quant(w_bits=1, a_bits=1):
    return ResNet_quant(BasicBlock_quant, [3,4,6,3], w_bits=w_bits, a_bits=a_bits)

def ResNet50_quant(w_bits=1, a_bits=1):
    return ResNet_quant(Bottleneck_quant, [3,4,6,3], w_bits=w_bits, a_bits=a_bits)

def ResNet101_quant(w_bits=1, a_bits=1):
    return ResNet_quant(Bottleneck_quant, [3,4,23,3], w_bits=w_bits, a_bits=a_bits)

def ResNet152_quant(w_bits=1, a_bits=1):
    return ResNet_quant(Bottleneck_quant, [3,8,36,3], w_bits=w_bits, a_bits=a_bits)


def test():
    net = ResNet18_quant()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
