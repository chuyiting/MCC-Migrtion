'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCSeg.py

    \brief Definition of the network architecture MCSeg for  
           segmentation tasks.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import math
import sys
import os
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import MLP_2_hidden, batch_norm_RELU_drop_out, conv_1x1

def create_network(points, batchIds, features, catLabels, numInputFeatures, numCats, numParts, batchSize, k, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    ############################################  Compute point hierarchy
    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.025, 0.1, 0.4], "MCSeg_PH", batchSize)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25)

    ############################################ Encoder

    # First Convolution
    convFeatures1 = mConvBuilder.create_convolution(
        convName="Conv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k,
        convRadius=0.03,
        multiFeatureConv=True)
    
    # First Pooling
    bnConvFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_In_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures1 = conv_1x1("Reduce_Pool_1", bnConvFeatures1, k, k*2)
    bnConvFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_Out_BN", bnConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    poolFeatures1 = mConvBuilder.create_convolution(
        convName="Pool_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        outPointLevel=1, 
        inFeatures=bnConvFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.05,
        KDEWindow= 0.2)

    # Second Convolution
    bnPoolFeatures1 = batch_norm_RELU_drop_out("Conv_2_In_BN", poolFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=bnPoolFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.1)
    convFeatures2 = tf.concat([poolFeatures1, convFeatures2], 1)
    
    # Second Pooling
    bnConvFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_In_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures2 = conv_1x1("Reduce_Pool_2", bnConvFeatures2, k*4, k*4)
    bnConvFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_Out_BN", bnConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    poolFeatures2 = mConvBuilder.create_convolution(
        convName="Pool_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=bnConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.2,
        KDEWindow= 0.2)
    
    # Third Convolution
    bnPoolFeatures2 = batch_norm_RELU_drop_out("Conv_3_In_BN", poolFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = mConvBuilder.create_convolution(
        convName="Conv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=bnPoolFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.4)
    convFeatures3 = tf.concat([poolFeatures2, convFeatures3], 1)

    # Third Pooling
    bnConvFeatures3 = batch_norm_RELU_drop_out("Reduce_Pool_3_In_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures3 = conv_1x1("Reduce_Pool_3", bnConvFeatures3, k*8, k*8)
    bnConvFeatures3 = batch_norm_RELU_drop_out("Reduce_Pool_3_Out_BN", bnConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    poolFeatures3 = mConvBuilder.create_convolution(
        convName="Pool_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=3, 
        inFeatures=bnConvFeatures3,
        inNumFeatures=k*8, 
        convRadius=0.8,
        KDEWindow= 0.2)
    
    # Fourth Convolution
    bnPoolFeatures3 = batch_norm_RELU_drop_out("Conv_4_In_BN", poolFeatures3, isTraining, useConvDropOut, keepProbConv)
    convFeatures4 = mConvBuilder.create_convolution(
        convName="Conv_4", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3, 
        inFeatures=bnPoolFeatures3,
        inNumFeatures=k*8, 
        convRadius=math.sqrt(3.0)+0.1)
    convFeatures4 = tf.concat([poolFeatures3, convFeatures4], 1)

    
    ############################################ Decoder
    
    # Third upsampling
    bnConvFeatures4 = batch_norm_RELU_drop_out("Up_3_4_BN", convFeatures4, isTraining, useConvDropOut, keepProbConv)
    upFeatures3_4 = mConvBuilder.create_convolution(
        convName="Up_3_4", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3,
        outPointLevel=2, 
        inFeatures=bnConvFeatures4,
        inNumFeatures=k*16, 
        convRadius=math.sqrt(3.0)+0.1)
    deConvFeatures3 = tf.concat([upFeatures3_4, convFeatures3], 1)
    deConvFeatures3 = batch_norm_RELU_drop_out("DeConv_3_Reduce_In_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures3 = conv_1x1("DeConv_3_Reduce", deConvFeatures3, k*24, k*8)
    deConvFeatures3 = batch_norm_RELU_drop_out("DeConv_3_Reduce_Out_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures3 = mConvBuilder.create_convolution(
        convName="DeConv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=deConvFeatures3,
        inNumFeatures=k*8, 
        convRadius=0.4)   
    
    # Second upsampling
    bnDeConvFeatures3 = batch_norm_RELU_drop_out("Up_2_3_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    upFeatures2_3 = mConvBuilder.create_convolution(
        convName="Up_2_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2,
        outPointLevel=1, 
        inFeatures=bnDeConvFeatures3,
        inNumFeatures=k*8, 
        convRadius=0.2)
    deConvFeatures2 = tf.concat([upFeatures2_3, convFeatures2], 1)
    deConvFeatures2 = batch_norm_RELU_drop_out("DeConv_2_Reduce_In_BN", deConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures2 = conv_1x1("DeConv_2_Reduce", deConvFeatures2, k*12, k*4)
    deConvFeatures2 = batch_norm_RELU_drop_out("DeConv_2_Reduce_Out_BN", deConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures2 = mConvBuilder.create_convolution(
        convName="DeConv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=deConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.1)    
    
    # First multiple upsamplings
    bnDeConvFeatures2 = batch_norm_RELU_drop_out("Up_1_2_BN", deConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    upFeatures1_2 = mConvBuilder.create_convolution(
        convName="Up_1_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1,
        outPointLevel=0, 
        inFeatures=bnDeConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.05)
    bnDeConvFeatures3 = batch_norm_RELU_drop_out("Up_1_3_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    upFeatures1_3 = mConvBuilder.create_convolution(
        convName="Up_1_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2,
        outPointLevel=0, 
        inFeatures=bnDeConvFeatures3,
        inNumFeatures=k*8, 
        convRadius=0.2)    
    deConvFeatures1 = tf.concat([upFeatures1_2, upFeatures1_3, convFeatures1], 1)
    deConvFeatures1 = batch_norm_RELU_drop_out("DeConv_1_Reduce_In_BN", deConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures1 = conv_1x1("DeConv_1_Reduce", deConvFeatures1, k*13, k*4)
    deConvFeatures1 = batch_norm_RELU_drop_out("DeConv_1_Reduce_Out_BN", deConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures1 = mConvBuilder.create_convolution(
        convName="DeConv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=deConvFeatures1,
        inNumFeatures=k*4, 
        convRadius=0.03)  
    
    
    # Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_hier_final", deConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    #Convert cat labels
    catLabelOneHot = tf.one_hot(catLabels, numCats, on_value=1.0, off_value=0.0)
    catLabelOneHot = tf.reshape(catLabelOneHot, [-1, numCats])
    finalInput = tf.concat([catLabelOneHot, finalInput], 1)
    finalLogits = MLP_2_hidden(finalInput, k*4 + numCats, k*4, k*2, numParts, "Final_Logits", keepProbFull, isTraining, useDropOutFull, useInitBN = False)

    return finalLogits
