"""
This code is generated by Ridvan Salih KUZU @UNIROMA3
LAST EDITED:  02.03.2020
ABOUT SCRIPT:
It is a script to compare vectors in different distance metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullPairComparer(nn.Module):

    def __init__(self, comparison='cosine'):
        super(FullPairComparer, self).__init__()
        if comparison=='euclidian':
            self.pdist = EuclidianDistance()
        elif comparison=='pearson':
            self.pdist = PearsonCorrelation()
        elif comparison=='cosine':
            self.pdist = CosineDistance()
        elif comparison=='norm':
            self.pdist=NormDistance()
        elif comparison=='hamming':
            self.pdist=HammingDistance()

    def forward(self, embeddings_1, embeddings_2):
        embeddings_1 = embeddings_1.cuda()
        embeddings_2 = embeddings_2.cuda()
        distances = self.pdist.forward(embeddings_1, embeddings_2)
        return distances


class HammingDistance(nn.Module):
    def __init__(self):
        super(HammingDistance, self).__init__()

    def forward(self, x1, x2):
        x = ~torch.eq(x1, x2)
        h = torch.mean(x.type(torch.FloatTensor),1)
        return h

class NormDistance(nn.Module):
    def __init__(self):
        super(NormDistance, self).__init__()

    def forward(self, x1, x2):
        return torch.dist(x1,x2,2)

class PearsonCorrelation(nn.Module):

    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, x1, x2):
        return self.pearsonr(x1,x2)

    def pearsonr(self,x, y):

        mean_x = torch.mean(x,1).unsqueeze_(-1).expand_as(x)
        mean_y = torch.mean(y,1).unsqueeze_(-1).expand_as(y)
        xm = (x-mean_x)
        ym = (y-mean_y)
        sim=F.cosine_similarity(xm,ym)
        return 1-sim

class EuclidianDistance(nn.Module):

    def __init__(self):
        super(EuclidianDistance, self).__init__()

    def forward(self, x1, x2):
        return self.euclidian(x1,x2)

    def normalize(self,vectors):
        qn = torch.norm(vectors, p=2, dim=1).detach()
        norm_vec = vectors.div(qn.view(-1, 1).expand_as(vectors))
        return norm_vec
    def euclidian(self,x, y):

        #nx=self.normalize(x)
        #ny=self.normalize(y)
        sim=F.pairwise_distance(x,y)
        return sim

class CosineDistance(nn.Module):

    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, x1, x2):
        return self.cosinedist(x1,x2)

    def cosinedist(self,x, y):
        sim=F.cosine_similarity(x,y)
        return 1-sim


def feature_binarizer(data):
    return torch.sign(F.relu(data))

