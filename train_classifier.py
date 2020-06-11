from models.CPC import CPC
from models.MobileNetV2_Encoder import MobileNetV2_Encoder
from models.mobileNetV2 import MobileNetV2
from models.Resnet_Encoder import ResNet_Encoder
from models.Resnet import ResNet
from data.data_handlers import get_stl10_dataloader
from argparser.train_classifier_argparser import argparser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Process a batch, return accuracy and loss
def fwd_pass(X, y, train=False):
    # Run the network
    if train:
        net.zero_grad()
        net.train()
        outputs = net(X)

    if not train:
        net.eval()
        with torch.no_grad():
            outputs = net(X)

    # Compute accuracy
    matches = [torch.argmax(i) == j for i, j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)

    # Compute loss
    loss = loss_function(outputs, y) 

    if train:
        loss.backward()
        optimizer.step()

    return loss, acc 

# Train net
def train():
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):

        for batch_img, batch_lbl in tqdm(train_loader, dynamic_ncols=True):
            loss, acc = fwd_pass(batch_img.to(args.device), batch_lbl.to(args.device), train=True)    

        test_loss, test_acc = test()

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        print(f"Epoch: {epoch}/{args.epochs}\n"
               f"Train: {round(float(loss),4)}, {round(float(acc*100), 2)}%\n"
                f"Test:  {round(float(test_loss),4)}, {round(float(test_acc*100), 2)}%")

        scheduler.step()
        
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
    
    print(f"Best Accuracy: {round(float(best_acc),4)} - epoch {best_epoch}")


# Process test data to find test loss/accuracy
def test():
    total_test_acc = 0
    total_test_loss = 0

    # Process all of the test data
    for batch_img, batch_lbl in tqdm(test_loader, dynamic_ncols=True):
        loss, acc = fwd_pass(batch_img.to(args.device), batch_lbl.to(args.device))  
        total_test_acc += acc
        total_test_loss += loss

    return total_test_loss / len(test_loader), total_test_acc / len(test_loader)

if __name__ == "__main__":
    args = argparser()
    print(f"Running on {args.device}")

    # Get selected dataset
    if args.dataset == "stl10":
        _, _, train_loader, _, test_loader, _ = get_stl10_dataloader(args.batch_size, labeled=True)
    elif args.dataset == "cifar10":
        raise NotImplementedError
    elif args.dataset == "cifar100":
        raise NotImplementedError
    else:
        raise Exception("Invalid Argument")

    # Define network and optimizer for given train_selection
    if args.train_selection == 0:
        print("Training CPC Classifier")

        if args.model_num == -1:
            raise Exception("For Training CPC model_num needs to be set")

        # Load the CPC trained encoder (with classifier layer activated)
        if args.encoder in ("resnet34", "resnet50") :
            net = ResNet_Encoder(args, use_classifier=True).to(args.device)
        elif args.encoder == "mobielnetV2":
            net = MobileNetV2_Encoder(args, use_classifier=True).to(args.device)
        else:
            raise Exception("Invalid Argument")
        
        encoder_path = f"TrainedModels/{args.dataset}/trained_encoder"
        net.load_state_dict(torch.load(f"{encoder_path}_{args.encoder}_{args.model_num}.pt"))        
        net = net.to(args.device)

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    elif args.train_selection == 1:
        print("Training Fully Supervised")

        # Load the network        
        if args.encoder in ("resnet34", "resnet50"):
            net = ResNet(args).to(args.device)
        elif args.encoder == "mobilenetV2":
            net = MobileNetV2(num_classes=args.num_classes).to(args.device)
        else:
            raise Exception("Invalid Argument")

        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # Train network
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.1)
    loss_function = nn.NLLLoss()

    try:
        train()
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt - Should save things")

    





    