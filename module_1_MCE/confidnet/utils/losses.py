import torch
import torch.nn as nn
import torch.nn.functional as F

from confidnet.utils import misc


class SelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self, config_args, device):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.weighting = config_args["training"]["loss"]["weighting"]
        self.device = device
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input[0], dim=1)
        confidence = torch.sigmoid(input[1]).squeeze()
        # Apply optional weighting
        weights = torch.ones_like(target).type(torch.FloatTensor).to(self.device)

        print('loss_weights:',weights,'loss_conf',confidence,'los_probs',probs,'loss_target',target,target.type,target.shape)

        weights[(probs.argmax(dim=1) != target)] *= self.weighting # weighting = default
        labels_hot = misc.one_hot_embedding(target, self.nb_classes).to(self.device)

        print('loss_hot:',labels_hot)
        print('loss_w1:',weights)

        loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
        return torch.mean(loss)





# PYTORCH LOSSES LISTS
PYTORCH_LOSS = {"cross_entropy": nn.CrossEntropyLoss}

# CUSTOM LOSSES LISTS
CUSTOM_LOSS = {
    "selfconfid_mse": SelfConfidMSELoss

}
