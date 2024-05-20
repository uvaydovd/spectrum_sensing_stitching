import torch
import numpy as np
from src.models import U_Net
from src.handler import TrainValHandler
from matplotlib import pyplot as plt
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ts','--TrainSet',type=str,metavar='',default='./train.h5',
                        help='filepath of training set')
    parser.add_argument('-vs','--ValSet',type=str,metavar='',default='./test.h5',
                        help='filepath of validation set')
    parser.add_argument('-d','--Device',type=int,default=-1,metavar='',
                        help='specify the device for running model')
    return parser.parse_args()

def train():
    args = arg_parser()
    trainset = args.TrainSet
    valset = args.ValSet
    model = U_Net(2,5,is_attention=True,alpha=1,beta=5)
    if args.Device == -1:
        device = 'cpu'
    else:
        device = 'cuda:%d'%args.Device
    batchsize = 256
    lr=1e-3
    epochs=100
    pt=30
    ckpt = 'multilabel.pth'
    handler = TrainValHandler(model,device,trainset,valset,ckpt,lr=lr,epochs=epochs,patience=pt,batchsize=batchsize)
    history = handler.train()

    # retrieve the best model and convert to onnx
    model = torch.load(ckpt,map_location='cpu')
    model.eval()
    dummy_input = torch.randn(1, 2, 1024, requires_grad=True)  
    torch_out = model(dummy_input)
    torch.onnx.export(model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            "multilabel.onnx",       # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput'], # the model's output names 
            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 


if __name__ == '__main__':
    train()