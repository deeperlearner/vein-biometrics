"""
This code is generated by Ridvan Salih KUZU @UNIROMA3
LAST EDITED:  26.02.2020
ABOUT SCRIPT:
It is a script for definining CNN model which is modified for better feature representation.
"""
import torch
import torch.nn as nn
from torchvision.models import densenet161, resnext101_32x8d, mnasnet1_0
import torch.nn.functional as F


class DenseNet161_Modified(nn.Module):
    """
        EMBEDDING_SIZE: It is the dimension of the embedding feature vector (i.e. 1024).
        CLASS_SIZE: Total number of classes in training and validation (i.e. 100),
                    it is compulsary if the ONLY_EMBEDDINGS is FALSE.
        PRETRAINED: TRUE or FALSE to flag the Imagenet pretrained model.
        ONLY_EMBEDDINGS: if TRUE, dimensions of both training/validation and test outputs are equal to EMBEDDING SIZE.
                         if FALSE, training/validation outputs = CLASS_SIZE while test outputs = EMBEDDING SIZE.
                         TRUE is necessary if custom penalty functions are utilized such as AAMP, LMCP
                         FALSE is necessary if standard softmax is preferred for training.
        L2_NORMED: The output of test pairs are normalized if it is TRUE. For AAMP, LMCP (Margin losses) it should be TRUE

    """
    def __init__(self, embedding_size, class_size=None, pretrained=True, only_embeddings=True, l2_normed=True):
        super(DenseNet161_Modified, self).__init__()
        self.only_embeddings=only_embeddings
        self.l2_normed=l2_normed
        self.model = densenet161(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(2208),
            nn.Dropout(p=0.5),
            nn.Linear(2208, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        if not self.only_embeddings:
            self.final=nn.Linear(embedding_size,class_size)

        # Weight initialization
        for m in self.model.classifier:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, train=True):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        if train:
            if self.only_embeddings:
                return self.model.classifier(out)
            else:
                return self.final(self.model.classifier(out))
        else:
            if self.l2_normed:
                return l2_norm(self.model.classifier(out))
            else:
                return self.model.classifier(out)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Resnext101_32x8d_Modified(nn.Module):
    """
    Wraps a base model with embedding layer modifications.
    """
    def __init__(self, embedding_size, class_size=None, pretrained=True, only_embeddings=True,l2_normed=True):
        super(Resnext101_32x8d_Modified, self).__init__()
        self.only_embeddings = only_embeddings
        self.l2_normed = l2_normed
        basemodel = resnext101_32x8d(pretrained=pretrained)

        model = nn.Sequential(
            basemodel.conv1,
            basemodel.bn1,
            basemodel.relu,
            basemodel.maxpool,

            basemodel.layer1,
            basemodel.layer2,
            basemodel.layer3,
            basemodel.layer4,

            basemodel.avgpool
        )
        model.name = "resnext101_32x8d"

        self.feature = model

        self.embedder = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.final = None
        if not only_embeddings:
            self.final = nn.Linear(embedding_size, class_size, bias=False)

    def forward(self, images,train=True):
        x = self.feature(images)
        x = torch.flatten(x, 1)

        if train:
            if self.only_embeddings:
                return self.embedder(x)
            else:
                return self.final(self.embedder(x))
        else:
            if self.l2_normed:
                return l2_norm(self.embedder(x)) #TODO: F.normalize(outs, p=2, dim=1)
            else:
                return self.embedder(x)

class MNASNet_Modified(nn.Module):
    """
        EMBEDDING_SIZE: It is the dimension of the embedding feature vector (i.e. 1024).
        CLASS_SIZE: Total number of classes in training and validation (i.e. 100),
                    it is compulsary if the ONLY_EMBEDDINGS is FALSE.
        PRETRAINED: TRUE or FALSE to flag the Imagenet pretrained model.
        ONLY_EMBEDDINGS: if TRUE, dimensions of both training/validation and test outputs are equal to EMBEDDING SIZE.
                         if FALSE, training/validation outputs = CLASS_SIZE while test outputs = EMBEDDING SIZE.
                         TRUE is necessary if custom penalty functions are utilized such as AAMP, LMCP
                         FALSE is necessary if standard softmax is preferred for training.
        L2_NORMED: The output of test pairs are normalized if it is TRUE. For AAMP, LMCP (Margin losses) it should be TRUE

    """
    def __init__(self, embedding_size, class_size=None, pretrained=True, only_embeddings=True, l2_normed=True):
        super(MNASNet_Modified, self).__init__()
        self.only_embeddings = only_embeddings
        self.l2_normed = l2_normed
        self.model = mnasnet1_0(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Dropout(p=0.5),
            nn.Linear(1280, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        if not self.only_embeddings:
            self.final = nn.Linear(embedding_size, class_size)

        # Weight initialization
        for m in self.model.classifier:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, train=True):
        x = rgb_to_grayscale(x, num_output_channels=3)
        out = self.model.layers(x)
        out = out.mean([2, 3])
        if train:
            if self.only_embeddings:
                return self.model.classifier(out)
            else:
                return self.final(self.model.classifier(out))
        else:
            if self.l2_normed:
                return l2_norm(self.model.classifier(out))
            else:
                return self.model.classifier(out)


def rgb_to_grayscale(img, num_output_channels: int = 1):
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")
    # _assert_channels(img, [1, 3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    # r, g, b = img.unbind(dim=-3)
    r, g, b = img.split(1, dim=-3)
    r, g, b = r.squeeze(dim=-3), g.squeeze(dim=-3), b.squeeze(dim=-3)  # Remove the channel dimension
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


class PalmCNN(nn.Module):
    def __init__(self, embedding_size, bn_momentum: float = 0.1, dropout: float = 0.2):
        super(PalmCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 16, stride=4),
            nn.BatchNorm2d(32, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 7, stride=2),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def forward(self, x, train=True):
        x = rgb_to_grayscale(x)
        x = self.model(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MNASNet_Modified(512)
    # print(model.model)
    # print(model.model.layers)
    # c = count_parameters(model)
    # print(c)
    model = PalmCNN(512)
    x = torch.rand(8, 3, 128, 128)
    out = model(x)
    print(out)
