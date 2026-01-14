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
        
        # 1. ノイズ x_0 を生成 (標準正規分布)
        x_0 = torch.randn_like(x_1)
        
        # 2. コスト行列の計算とペアリング (CPUで計算)
        #    目的: dot product (x^T y) を最大化したい
        #    scipyのlinear_sum_assignmentは「最小化」を行うため、
        #    コスト行列を C = - (x^T y) として与える。
        with torch.no_grad():
            # 計算軽量化のためフラット化してCPUへ
            x_0_flat = x_0.view(b, -1).cpu()
            x_1_flat = x_1.view(b, -1).cpu()
            
            # 行列積で全ペアの内積を計算: (B, D) @ (D, B) -> (B, B)
            # Cost Matrix: C_ij = - (x_0[i] . x_1[j])
            cost_matrix = - torch.matmul(x_0_flat, x_1_flat.t()).numpy()
            
            # 線形割当問題を解く (Hungarian Algorithm)
            # row_ind[k] の要素と col_ind[k] の要素がペアになる
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # x_0 を並べ替えて x_1[i] とペアになるように整列させる
            # col_ind が x_1 のインデックスに対応するため、
            # x_1 の順番 (0, 1, ..., B) に合うように x_0 の行を持ってくる必要がある。
            
            # 例: x_1[0] とペアになる x_0 は？ -> col_indの中から 0 を探し、その時の row_ind を取得
            # argsort(col_ind) を使うと、x_1 のインデックス順 (0, 1, 2...) に対応する row_ind の並びが得られる
            sort_idx = np.argsort(col_ind)
            
            # ペアリング後の x_0 (x_0_ordered[i] が x_1[i] とペアになる)
            x_0_ordered = x_0[row_ind[sort_idx]]

        # 3. 通常の Flow Matching Loss 計算 (整列済みペアを使用)
        t = torch.rand(b, device=device)
        
        # OT Path: x_t = (1 - (1-sigma)t) * x_0 + t * x_1
        x_t, target_v = self.sample_location_and_conditional_flow(x_0_ordered, x_1, t)
        
        # モデル予測
        pred_v = model(x_t, t, context=condition)
        
        # MSE Loss
        loss = torch.mean((pred_v - target_v) ** 2)
        
        return loss