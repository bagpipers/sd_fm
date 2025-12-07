import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union


class SimpleODESolver:
    """
    サンプリング（推論）のための簡易的なODEソルバー。
    dx/dt = v_t(x, t, condition) (Eq. 1) を数値的に解きます。
    ★ CFG (Classifier-Free Guidance) に対応 ★
    """
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def sample(
        self, 
        x_init: torch.Tensor, 
        condition: torch.Tensor, 
        # --- ★ 引数を追加 ---
        uncond_condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        # --- ★ 追加ここまで ---
        steps: int = 25, 
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        x_init (ノイズ, t=0) から出発し、t=1 まで積分して画像を生成します。

        Args:
            x_init (torch.Tensor): 初期ノイズ (N(0, I))
            condition (torch.Tensor): 条件付き埋め込み (cond)
            uncond_condition (Optional[torch.Tensor]): 無条件埋め込み (uncond)
                None の場合、CFG は実行されません (guidance_scale=0 と同等)
            guidance_scale (float): CFGの強度。
                0.0 の場合、CFG は実行されません。
            steps (int): 積分ステップ数
            method (str): 'euler' または 'midpoint'

        Returns:
            torch.Tensor: 生成された画像 (x_1)
        """
        b = x_init.shape[0]
        device = x_init.device
        
        # CFGが有効かどうかのフラグ
        use_cfg = (guidance_scale > 0.0) and (uncond_condition is not None)
        
        if not use_cfg and guidance_scale > 0.0:
            print("Warning: guidance_scale > 0.0 but uncond_condition is None. Disabling CFG.")
            use_cfg = False
            
        time_steps = torch.linspace(0, 1, steps + 1, device=device)
        dt = time_steps[1] - time_steps[0]
        
        x = x_init
        
        for i in range(steps):
            t = time_steps[i]
            t_batch = torch.ones(b, device=device) * t
            
            # --- ★ 修正点: CFGに対応した速度場の予測 ---
            
            if method == 'euler':
                # オイラー法: x_{t+dt} = x_t + v_t * dt
                
                # 速度場 v_t を計算
                v_t = self._get_velocity(x, t_batch, condition, uncond_condition, use_cfg, guidance_scale)
                x = x + v_t * dt
                
            elif method == 'midpoint':
                # 中点法 (2次精度)
                
                # k1 = v_t(x_t)
                v_t1 = self._get_velocity(x, t_batch, condition, uncond_condition, use_cfg, guidance_scale)
                x_mid = x + v_t1 * dt * 0.5
                
                t_mid = t + dt * 0.5
                t_mid_batch = torch.ones(b, device=device) * t_mid
                
                # k2 = v_{t+dt/2}(x_t + k1*dt/2)
                v_t2 = self._get_velocity(x_mid, t_mid_batch, condition, uncond_condition, use_cfg, guidance_scale)
                
                # x_{t+dt} = x_t + k2 * dt
                x = x + v_t2 * dt
            # --- ★ 修正ここまで ---
                
        return x

    def _get_velocity(
        self, x: torch.Tensor, t: torch.Tensor, 
        cond: torch.Tensor, uncond: Optional[torch.Tensor], 
        use_cfg: bool, guidance_scale: float
    ) -> torch.Tensor:
        """ 
        CFGを適用した速度場 v_t を計算するヘルパー関数 
        """
        
        # 常に条件付きの予測は計算
        v_cond = self.model(x, t, context=cond)
        
        if use_cfg:
            # 無条件の予測も計算
            v_uncond = self.model(x, t, context=uncond)
            
            # v_t = v_uncond + scale * (v_cond - v_uncond)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            # CFGを使わない場合
            return v_cond