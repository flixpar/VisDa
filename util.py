import torch
from torch import nn
import torch.nn.functional as F

class CrossEntropyLoss2dAlt(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        print(inputs.size())
        print(F.log_softmax(inputs).size())
        print(self.softmax(inputs).size())
        print()
        # return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return self.nll_loss(self.softmax(inputs), targets)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class CrossEntropyLoss2dMax(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=None):
        super(CrossEntropyLoss2dMax, self).__init__()
        #self.softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        #return self.nll_loss(self.softmax(inputs), targets)
        #print(F.log_softmax(inputs, dim=1).size())
        #print(torch.max(F.log_softmax(inputs), 1).size())
        print(torch.max(F.log_softmax(inputs, dim=1), 1)[0].size())
        print(targets.size())
        print()
        return self.nll_loss(torch.max(F.log_softmax(inputs, dim=1), 1)[0], targets)

def cross_entropy2d(input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
        if size_average:
            loss /= mask.data.sum()
        return loss
