"""预设超参配置

提供若干命名预设，便于快速复现实验或作为默认最佳配置。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Preset:
    name: str
    params: dict


PRESETS = {
    # 当前固定窗口最佳（基于最新 summary_window.csv）
    'window_best_v1': Preset(
        name='window_best_v1',
        params=dict(
            strategies=['window'],
            window_size=3,
            volatility_mode='suppress',
            weight_trend=1.0,
            weight_zloss=0.5,
            weight_vol=0.5,
            max_repeats=3,
        ),
    ),
    # 稳定备选
    'window_stable_v1': Preset(
        name='window_stable_v1',
        params=dict(
            strategies=['window'],
            window_size=5,
            volatility_mode='suppress',
            weight_trend=0.5,
            weight_zloss=1.0,
            weight_vol=1.0,
            max_repeats=3,
        ),
    ),
    # 自适应窗口的默认最佳（在自适应四组合中最佳）
    'window_adaptive_v1': Preset(
        name='window_adaptive_v1',
        params=dict(
            strategies=['window'],
            window_size=5,
            volatility_mode='suppress',
            adaptive_window=True,
            window_small=3,
            window_large=9,
            adapt_high_action='shrink',
            adapt_low_action='expand',
            max_repeats=3,
        ),
    ),
}


def apply_preset(args) -> None:
    """原地修改 argparse.Namespace 根据预设名覆盖参数。

    若不存在该预设名，不做修改。
    """
    name = getattr(args, 'preset', None)
    if not name:
        return
    preset = PRESETS.get(name)
    if not preset:
        return
    for k, v in preset.params.items():
        setattr(args, k, v)


