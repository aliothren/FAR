import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, x1, x2):
        return 1 - self.cos_sim(x1, x2).mean()


class CombinedLoss(nn.Module):
    def __init__(self, cos_weight= 0.6):
        super(CombinedLoss, self).__init__()
        self.cos_loss = CosineSimilarityLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.cos_weight = cos_weight
        self.cls_weight = 1 - self.cos_weight
        
    def forward(self, x1, x2, logits, labels):
        cos_loss = self.cos_loss(x1, x2)
        cls_loss = self.cls_loss(logits, labels)
        combined_loss = self.cos_weight * cos_loss + self.cls_weight * cls_loss
        return combined_loss