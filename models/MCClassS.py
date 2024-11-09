'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCClassS.py

    \brief Definition of the network architecture MCClassS for classification 
           tasks.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import torch
import torch.nn as n
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.MCConvBuilder import PointHierarchy, ConvolutionBuilder
from utils.MCNetworkUtils import MLP2Hidden, BatchNormReLUDropout, Conv1x1

import torch
import torch.nn as nn
import math

class MCClassS(nn.Module):
    def __init__(self, numInputFeatures, k, numOutCat, batch_size, keepProbConv=0.8, keepProbFull=0.8,
                 useConvDropOut=False, useDropOutFull=True):
        super(MCClassS, self).__init__()
        self.k = k
        self.numOutCat = numOutCat
        self.batch_size = batch_size
        self.useConvDropOut = useConvDropOut
        self.useDropOutFull = useDropOutFull

        self.conv1 = ConvolutionBuilder(KDEWindow=0.2, convName = "Conv_1", inPointLevel=0, outPointLevel=1, inNumFeatures=numInputFeatures, outNumFeatures=k, convRadius= 0.2, multiFeatureConvs=True)
        self.conv2 = ConvolutionBuilder(KDEWindow=0.2, convName = "Conv_2", inPointLevel=1, outPointLevel=2, inNumFeatures=k*2, convRadius= 0.8)
        self.conv3 = ConvolutionBuilder(KDEWindow=0.2, convName = "Conv_3", inPointLevel=2, outPointLevel=3, inNumFeatures=k*4, convRadius=math.sqrt(3.0)+0.1)
        
        # Convolutional layers
        self.bn_relu_dropout1 = BatchNormReLUDropout(k, use_dropout=useConvDropOut, keep_prob=keepProbConv)
        self.conv1x1_1 = Conv1x1(k, k * 2)
        self.bn_relu_dropout2 = BatchNormReLUDropout(k * 2, use_dropout=useConvDropOut, keep_prob=keepProbConv)
        self.conv1x1_2 = Conv1x1(k * 2, k * 4)
        self.bn_relu_dropout3 = BatchNormReLUDropout(k * 4, use_dropout=useConvDropOut, keep_prob=keepProbConv)

        # Fully connected MLP
        self.final_bn_relu_dropout = BatchNormReLUDropout(k * 4, use_dropout=useConvDropOut, keep_prob=keepProbConv)
        self.final_mlp = MLP2Hidden(k * 4, k * 2, k, numOutCat, use_dropout=useDropOutFull, keep_prob=keepProbFull)

    def forward(self, points, batch_ids, features):
        ############################################ Compute point hierarchy
        # Initialize PointHierarchy, we are creating hierarchy every time!
        ############################################ Compute point hierarchy
        mPointHierarchy = PointHierarchy(points, features, batch_ids, [0.1, 0.4, math.sqrt(3.0)+0.1], "MCClassS_PH", self.batch_size)

        ############################################ Convolutions
        # Convolution 1
        convFeatures1 = self.conv1(mPointHierarchy, features)

        # BatchNorm + ReLU + Dropout + Conv1x1 for Convolution 1
        convFeatures1 = self.bn_relu_dropout1(convFeatures1)
        convFeatures1 = self.conv1x1_1(convFeatures1)
        convFeatures1 = self.bn_relu_dropout2(convFeatures1)

        # Convolution 2
        convFeatures2 = self.conv2(mPointHierarchy, convFeatures1)

        # BatchNorm + ReLU + Dropout + Conv1x1 for Convolution 2
        convFeatures2 = self.bn_relu_dropout2(convFeatures2)
        convFeatures2 = self.conv1x1_2(convFeatures2)
        convFeatures2 = self.bn_relu_dropout3(convFeatures2)

        # Convolution 3
        convFeatures3 = self.conv3(mPointHierarchy, convFeatures2)
        print(f'conv3 shape: {convFeatures3.shape}')

        # Fully connected MLP for final global features
        finalInput = self.final_bn_relu_dropout(convFeatures3)
        finalLogits = self.final_mlp(finalInput)
        

        return finalLogits

