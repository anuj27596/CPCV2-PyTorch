# Based on:
# https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm2d(planes, norm):
    if norm == "none":
        return nn.Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(planes)
    elif norm == "layer":
        return nn.GroupNorm(1, planes)
    elif norm == "instance":
        return nn.GroupNorm(planes, planes)
    else:
        raise Exception("Undefined norm choice")


class wide_basic(nn.Module):
    def __init__(self, args, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()

        # If there isn't normalisation then the conv layers need biasing
        bias = True if ( args.norm == "none" ) else False

        self.norm1 = norm2d(in_planes, args.norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm2 = norm2d(planes, args.norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=bias),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.norm1(x))))
        out = self.conv2(F.relu(self.norm2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet_Encoder(nn.Module):
    def __init__(self, args, depth, widen_factor, use_classifier, dropout_rate=0, input_channels=1):
        super(Wide_ResNet_Encoder, self).__init__()
        self.in_planes = 16
        self.patch_size = args.patch_size
        self.use_classifier = use_classifier

        assert ((depth-4)%6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        # If there isn't normalisation then the conv layers need biasing
        bias = True if ( args.norm == "none" ) else False

        self.conv1 = nn.Conv2d(input_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=bias)
        self.layer1 = self._wide_layer(args, wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(args, wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(args, wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.norm1 = norm2d(nStages[3], args.norm)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(nStages[3], args.num_classes)

    def _wide_layer(self, args, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(args, self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        ### Convert image to patches
        # takes x as (batch_size, 1, 64, 64)
        # patches it to (batch_size, 7, 7, 1, 16, 16)
        # then flattens to (batch_size * 7 * 7, 1, 16, 16)
        x = (
            x.unfold(2, self.patch_size, self.patch_size // 2)
            .unfold(3, self.patch_size, self.patch_size // 2)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        n_patches_x = x.shape[1] # In general this will be 7
        n_patches_y = x.shape[2] # But it could be varied like in CPC V2
        x = x.view(
            x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
        )
        
        ### Run the model
        z = self.conv1(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = F.relu(self.norm1(z))
        z = self.avgpool(z)
        z = z.reshape(-1, n_patches_x, n_patches_y, z.shape[1]) # (batch_size, 7, 7, pred_size)

        ### Use classifier if specified
        if self.use_classifier:
            # Reshape z so that each image is seperate
            z = z.view(z.shape[0], n_patches_x * n_patches_y, z.shape[3])

            z = torch.mean(z, dim=1) # mean for each image, (batch_size, pred_size)
            z = self.classifier(z)
            z = F.log_softmax(z, dim=1)

        return z