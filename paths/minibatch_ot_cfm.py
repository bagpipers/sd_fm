import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from .ot_cfm import OTConditionalFlowMatching

class MinibatchOTCFM(OTConditionalFlowMatching):
    """
    Minibatch OT (Optimal Transport) Flow Matching 実装。
    
    学習バッチごとにノイズ x_0 を生成し、ターゲット画像 x_1 との間で
    内積 (x^T y) が最大となるようにペアリング（並べ替え）を行ってから
    Flow Matching の損失を計算します。
    """
    def compute_loss(self, model, x_1, condition):
        """
        Args:
            model: 学習対象のUNetモデル
            x_1: ターゲット画像データ (Batch, C, H, W)
            condition: 条件埋め込み (Batch, Context_Dim)
        """
        b = x_1.shape[0]
        device = x_1.device
        x_0 = torch.randn_like(x_1)
        with torch.no_grad():
            x_0_flat = x_0.view(b, -1).cpu()
            x_1_flat = x_1.view(b, -1).cpu()
            cost_matrix = - torch.matmul(x_0_flat, x_1_flat.t()).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            sort_idx = np.argsort(col_ind)
            x_0_ordered = x_0[row_ind[sort_idx]]
        t = torch.rand(b, device=device)
        x_t, target_v = self.sample_location_and_conditional_flow(x_0_ordered, x_1, t)
        pred_v = model(x_t, t, context=condition)
        loss = torch.mean((pred_v - target_v) ** 2)
        
        return loss