import torch
from .ot_cfm import OTConditionalFlowMatching 

class PairedOTCFM(OTConditionalFlowMatching):
    """
    SD-FM用に拡張したCFMクラス。
    内部でノイズを生成せず、外部から供給されたペア(x0, x1)を使用する。
    """
    def compute_loss_paired(self, model, x_0, x_1, condition):
        """
        Args:
            x_0: Paired Noise (SD-Loaderから供給)
            x_1: Target Data (SD-Loaderから供給)
            condition: Text Embedding
        """
        b = x_1.shape[0]
        device = x_1.device
        t = torch.rand(b, device=device)
        x_t, target_v = self.sample_location_and_conditional_flow(x_0, x_1, t)
        pred_v = model(x_t, t, context=condition)
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss