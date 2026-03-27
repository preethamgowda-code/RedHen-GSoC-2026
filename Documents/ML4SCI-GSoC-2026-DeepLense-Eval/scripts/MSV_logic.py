import torch
import torch.nn.functional as F

class MetacognitiveSupervisor:
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def get_msv_score(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        max_h = torch.log(torch.tensor(logits.shape[1], dtype=torch.float))
        msv_score = 1.0 - (entropy / max_h)
        return msv_score

    def validate_batch(self, logits):
        msv_scores = self.get_msv_score(logits)
        is_reliable = msv_scores > self.threshold
        return is_reliable, msv_scores
