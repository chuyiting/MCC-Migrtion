import sys
import math
import time
import argparse
import importlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.MCClassS import MCClassS
from utils.PyUtils import visualize_progress
from ModelNetDataSet import ModelNetDataSet

current_milli_time = lambda: time.time() * 1000.0


def create_loss(logits, labels, weight_decay, model):
    criterion = nn.CrossEntropyLoss()
    xentropy_loss = criterion(logits, labels)
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    reg_term = weight_decay * l2_reg
    return xentropy_loss, reg_term


def create_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def load_weights(model, optimizer):
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def compare_running_vs_batch_stats(model, points, batchIds, feature):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        print("Evaluation Mode - Using Running Statistics:")
        _ = model(points, batchIds, feature)  # Forward pass in eval mode to use running stats
        print_batchnorm_stats(model, mode="Running")

    model.train()  # Temporarily switch to train mode to get batch stats on test data
    with torch.no_grad():
        print("\nTrain Mode on Test Data - Using Batch Statistics:")
        _ = model(points, batchIds, feature)  # Forward pass in train mode to capture batch statistics
        print_batchnorm_stats(model, mode="Batch")

def print_batchnorm_stats(model, mode="Running"):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm1d):
            print(f"Layer: {name} - {mode} statistics")
            if mode == "Running":
                print(f"  Running mean: {layer.running_mean}")
                print(f"  Running variance: {layer.running_var}")
            elif mode == "Batch":
                print(f"  Batch mean: {layer.running_mean}")
                print(f"  Batch variance: {layer.running_var}")

def check_deterministic_outputs(model, points, batchIds, feature):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Run two forward passes
        output1 = model(points, batchIds, feature)
        output2 = model(points, batchIds, feature)
        # Check if outputs are identical
        if torch.allclose(output1, output2):
            print("Outputs are deterministic; dropout is correctly handled in eval mode.")
        else:
            print(torch.mean(output1 - output2))
            print("Outputs are not deterministic; dropout may still be active in eval mode.")

model_map = {
    'MCClassS' : MCClassS
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train MCCNN for classification of point clouds (ModelNet40)')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--model', default='MCClassS', help='model (default: MCClassS)')
    parser.add_argument('--grow', default=64, type=int, help='Grow rate (default: 64)')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size  (default: 32)')
    parser.add_argument('--maxEpoch', default=201, type=int, help='Max Epoch  (default: 201)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDecayFactor', default=0.5, type=float, help='Learning decay factor (default: 0.5)')
    parser.add_argument('--learningDecayRate', default=20, type=int, help='Learning decay rate  (default: 20 Epochs)')
    parser.add_argument('--maxLearningRate', default=0.00001, type=float, help='Maximum Learning rate (default: 0.00001)')
    parser.add_argument('--useDropOut', action='store_true', help='Use dropout (default: False)')
    parser.add_argument('--dropOutKeepProb', default=0.5, type=float, help='Keep neuron probability dropout (default: 0.5)')
    parser.add_argument('--useDropOutConv', action='store_true', help='Use dropout in convolution layers (default: False)')
    parser.add_argument('--dropOutKeepProbConv', default=0.8, type=float, help='Keep neuron probability dropout in convolution layers (default: 0.8)')
    parser.add_argument('--weightDecay', default=0.0, type=float, help='Weight decay (default: 0.0)')
    parser.add_argument('--nPoints', default=1024, type=int, help='Number of points (default: 1024)')
    parser.add_argument('--ptDropOut', default=0.95, type=float, help='Point dropout (default: 0.95)')
    parser.add_argument('--augment', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--nonunif', action='store_true', help='Train on non-uniform (default: False)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    parser.add_argument('--use_pretrain', default=False, action='store_true', help='whether to use pretrain weights')
    args = parser.parse_args()

    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp ../models/%s.py %s' % (args.model, args.logFolder))
    os.system('cp ModelNet.py %s' % (args.logFolder))
    logFile = args.logFolder + "/log.txt"

    with open(logFile, "a") as myFile:
        myFile.write(f"Model: {args.model}\n")
        myFile.write(f"Grow: {args.grow}\n")
        myFile.write(f"BatchSize: {args.batchSize}\n")
        myFile.write(f"MaxEpoch: {args.maxEpoch}\n")
        myFile.write(f"InitLearningRate: {args.initLearningRate}\n")
        myFile.write(f"LearningDecayFactor: {args.learningDecayFactor}\n")
        myFile.write(f"LearningDecayRate: {args.learningDecayRate}\n")
        myFile.write(f"MaxLearningRate: {args.maxLearningRate}\n")
        myFile.write(f"UseDropOut: {args.useDropOut}\n")
        myFile.write(f"DropOutKeepProb: {args.dropOutKeepProb}\n")
        myFile.write(f"UseDropOutConv: {args.useDropOutConv}\n")
        myFile.write(f"DropOutKeepProbConv: {args.dropOutKeepProbConv}\n")
        myFile.write(f"WeightDecay: {args.weightDecay}\n")
        myFile.write(f"nPoints: {args.nPoints}\n")
        myFile.write(f"ptDropOut: {args.ptDropOut}\n")
        myFile.write(f"Augment: {args.augment}\n")
        myFile.write(f"Nonunif: {args.nonunif}\n")

    print(f"Model: {args.model}")
    print(f"Grow: {args.grow}")
    print(f"BatchSize: {args.batchSize}")
    print(f"MaxEpoch: {args.maxEpoch}")
    print(f"InitLearningRate: {args.initLearningRate}")
    print(f"LearningDecayFactor: {args.learningDecayFactor}")
    print(f"LearningDecayRate: {args.learningDecayRate}")
    print(f"MaxLearningRate: {args.maxLearningRate}")
    print(f"UseDropOut: {args.useDropOut}")
    print(f"DropOutKeepProb: {args.dropOutKeepProb}")
    print(f"UseDropOutConv: {args.useDropOutConv}")
    print(f"DropOutKeepProbConv: {args.dropOutKeepProbConv}")
    print(f"WeightDecay: {args.weightDecay}")
    print(f"nPoints: {args.nPoints}")
    print(f"ptDropOut: {args.ptDropOut}")
    print(f"Augment: {args.augment}")
    print(f"Nonunif: {args.nonunif}")

    # Get train and test datasets
    allowedSamplingsTrain = []
    allowedSamplingsTest = []
    maxStoredPoints = int(float(args.nPoints) * (2.0 - args.ptDropOut))
    if args.nonunif:
        maxStoredPoints = 5000
        allowedSamplingsTrain = [0]
        allowedSamplingsTest = [0]
    else:
        allowedSamplingsTrain = [0]
        allowedSamplingsTest = [0]
    print('start loading dataset')
    mTrainDataSet = ModelNetDataSet(True, args.nPoints, args.ptDropOut, maxStoredPoints, args.batchSize, allowedSamplingsTrain, args.augment)
    mTestDataSet = ModelNetDataSet(False, args.nPoints, 1.0, maxStoredPoints, args.batchSize, allowedSamplingsTest, False)
    categories = mTrainDataSet.get_categories()
    print('finish loading dataset')
    
    #Create the network
    num_input_features = 1
    batch_size = args.batchSize
    k = args.grow
    num_out_cat = len(categories)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = model_map[args.model]
    model = model_class(numInputFeatures=num_input_features, k=k, numOutCat=num_out_cat, 
                                  batch_size=batch_size, keepProbConv=args.dropOutKeepProbConv, keepProbFull=args.dropOutKeepProb, 
                                  useConvDropOut=args.useDropOutConv, useDropOutFull=args.useDropOut).to(device)

    # TODO add learning rate decay per batch
    optimizer = optim.Adam(model.parameters(), lr=args.initLearningRate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learningDecayFactor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], args.maxLearningRate)

    start_epoch = 0
    if args.use_pretrain:
        print('use pretrained weights....')
        # Save the model's initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        model, optimizer, start_epoch = load_weights(model, optimizer)
        # Compare the parameters before and after loading the checkpoint
        for (name, initial_param), (name_after, loaded_param) in zip(initial_params.items(), model.named_parameters()):
            if not torch.equal(initial_param, loaded_param):
                print(f"Parameter '{name}' has been modified after loading checkpoint.")
            else:
                print(f"Parameter '{name}' matches after loading checkpoint.")
        
        print("Checkpoint loading test complete.")
            

    # Train model
    bestTestAccuracy = -1.0
    print("start training...")
    for epoch in range(start_epoch, args.maxEpoch):
        startEpochTime = current_milli_time()
        startTrainTime = current_milli_time()

        model.train()
        running_loss = 0.0
        total_accuracy = 0.0
        mTrainDataSet.start_iteration()
        num_iter = 0
        while mTrainDataSet.has_more_batches():
            _, points, batchIds, features, _, labels, _ = mTrainDataSet.get_next_batch()
            points = torch.from_numpy(points).float().cuda()
            batchIds = torch.from_numpy(batchIds).int().cuda()
            features = torch.from_numpy(features).float().cuda()
            labels = torch.from_numpy(labels).long().cuda()
            logits = model(points, batchIds, features)
            xentropy_loss, reg_term = create_loss(logits, labels, args.weightDecay, model)
            total_loss = xentropy_loss + reg_term
            running_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            accuracy = create_accuracy(logits, labels)
            total_accuracy += accuracy
            if num_iter % 50 == 0:
                print(f"B [{num_iter}], Loss: {total_loss.item():.4f}, Accuracy: {accuracy:.4f}")
            num_iter += 1
        
        endEpochTime = current_milli_time()   
        running_loss /= num_iter
        total_accuracy /= num_iter
        print(f"Epoch [{epoch + 1}/{args.maxEpoch}], Loss: {running_loss:.4f}, Accuracy: {total_accuracy:.4f}, Training time: {((endEpochTime-startEpochTime)/1000.0):.2f}")
        
        # Check on test data for early stopping
        if (epoch) % 10 == 0:
            model.eval()
            test_loss = 0.0
            test_accuracy = 0.0
            mTestDataSet.start_iteration()
            num_iter = 0
            with torch.no_grad():
                while mTestDataSet.has_more_batches():
                    num_iter += 1
                    _, points, batchIds, features, _, labels, _ = mTestDataSet.get_next_batch()
                    points = torch.from_numpy(points).float().cuda()
                    batchIds = torch.from_numpy(batchIds).int().cuda()
                    features = torch.from_numpy(features).float().cuda()
                    labels = torch.from_numpy(labels).long().cuda()
                    #check_deterministic_outputs(model, points, batchIds, features)
                    logits = model(points, batchIds, features)
                    xentropy_loss, reg_term = create_loss(logits, labels, args.weightDecay, model)
                    total_loss = xentropy_loss + reg_term
                    test_loss += total_loss.item()

                    accuracy = create_accuracy(logits, labels)
                    test_accuracy += accuracy
                    if num_iter % 10 == 0:
                        print(labels)
                        print(f'B[{num_iter}] test loss: {test_loss/ num_iter} accuracy: {test_accuracy / num_iter}')
               
            test_accuracy /= num_iter
            print(f"Test Accuracy: {test_accuracy:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }   
            torch.save(checkpoint, 'checkpoint.pth')
            print('saving model weights')

        if epoch % args.learningDecayRate == 0:
            lr_scheduler.step()

    print(f"Training completed. Best test accuracy: {bestTestAccuracy:.4f}")
