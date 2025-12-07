import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union


class OTConditionalFlowMatching:
    """
    Independent FM を用いた Conditional Flow Matching (CFM) 
    論文 "Flow Matching for Generative Modeling" (arXiv:2210.02747v2) 
    """
    def __init__(self, sigma_min: float = 1e-5):
        """
        Args:
            sigma_min (float): t=1 での標準偏差。ゼロに近い小さな値 (Eq. 20)。
        """
        self.sigma_min = sigma_min

    def sample_location_and_conditional_flow(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ノイズ x_0 (t=0) とデータ x_1 (t=1) から、時刻 t における
        経路上の点 x_t と、ターゲット速度 u_t を計算します。

        Args:
            x_0 (torch.Tensor): ノイズ (N(0, I) からサンプリング)
            x_1 (torch.Tensor): データ
            t (torch.Tensor): 時刻 (U[0, 1] からサンプリング)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (x_t, target_v)
                x_t: 時刻 t での補間されたサンプル
                target_v: モデルが学習すべきターゲット速度ベクトル
        """
        # t の次元を (B,) -> (B, 1, 1, 1) のように x の次元に合わせる
        t_batch = t.view(-1, *([1] * (x_1.dim() - 1)))
        
        # 1. OT Path の係数計算 
        # μ_t(x_1) = t * x_1
        # σ_t = 1 - (1 - σ_min) * t
        mu_t = t_batch * x_1
        sigma_t = 1 - (1 - self.sigma_min) * t_batch
        
        # 2. x_t のサンプリング 
        # x_t = σ_t * x_0 + μ_t(x_1)
        x_t = sigma_t * x_0 + mu_t
        
        # 3. ターゲット速度場 (Flow Matching Objective) 
        # u_t = d/dt [σ_t * x_0 + μ_t(x_1)]
        # u_t = d/dt [(1 - (1 - σ_min) * t) * x_0 + t * x_1]
        # u_t = -(1 - σ_min) * x_0 + x_1
        target_v = x_1 - (1 - self.sigma_min) * x_0
        
        return x_t, target_v

    def compute_loss(
        self, model: nn.Module, x_1: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        CFM (OT Path) の損失 を計算します。
        
        L_CFM = E[ || v_t(x_t, t, condition) - u_t ||^2 ]
        
        Args:
            model (nn.Module): 予測モデル v_t (UNet など)
            x_1 (torch.Tensor): データ (バッチ)
            condition (torch.Tensor): 条件 (テキスト埋め込みなど)

        Returns:
            torch.Tensor: スカラーの損失値
        """
        b, c, h, w = x_1.shape
        device = x_1.device
        
        # 1. ノイズ x_0 と時刻 t をランダムサンプリング
        # x_0 ~ p(x_0) = N(0, I)
        x_0 = torch.randn_like(x_1)
        # t ~ U[0, 1]
        t = torch.rand(b, device=device)
        
        # 2. x_t とターゲット速度 target_v (u_t) を計算 (OT Path)
        x_t, target_v = self.sample_location_and_conditional_flow(x_0, x_1, t)
        
        # 3. モデルによる速度予測 v_t
        # pred_v = v_t(x_t, t, condition)
        pred_v = model(x_t, t, context=condition)
        
        # 4. MSE Loss 
        # (pred_v - target_v) の二乗誤差の期待値 (バッチ平均)
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss