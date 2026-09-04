"""Project-level compatibility fixes for the IsaacLab SB3 wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab_rl.sb3 import Sb3VecEnvWrapper as _IsaacLabSb3VecEnvWrapper
from isaaclab_rl.sb3 import process_sb3_cfg as _process_sb3_cfg


def process_sb3_cfg(cfg: dict, num_envs: int) -> dict:
    """Process SB3 settings while allowing YAML ``null`` to disable value clipping."""

    cfg = dict(cfg)
    if cfg.get("clip_range_vf", ...) is None:
        cfg.pop("clip_range_vf")
    return _process_sb3_cfg(cfg, num_envs)


def vecnormalize_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the VecNormalize sidecar path used by SB3's checkpoint callback."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.name.startswith("model"):
        raise ValueError(f"Expected a model checkpoint filename, got {checkpoint_path.name!r}.")
    sidecar_name = checkpoint_path.name.replace("model", "model_vecnormalize", 1)
    return checkpoint_path.with_name(Path(sidecar_name).with_suffix(".pkl").name)


class Sb3VecEnvWrapper(_IsaacLabSb3VecEnvWrapper):
    """Convert terminal TensorBoard metrics to host scalars.

    IsaacLab environments keep termination terms on the simulation device.
    The base wrapper copies these values into SB3 ``info`` dictionaries
    unchanged, which makes SB3 append CUDA tensors to ``ep_success_buffer``.
    SB3 later applies NumPy reductions to that buffer and fails.  Only values
    for environments that actually terminated are converted here, so the
    normal GPU-side rollout path is otherwise unchanged.
    """

    _TERMINAL_METRIC_KEYS = (
        "success",
        "is_success",
        "collided",
        "time_out",
        "isolation_boundary",
    )

    @staticmethod
    def _to_host_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.numpy()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _process_extras(
        self,
        obs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        extras: dict,
        reset_ids: np.ndarray,
    ) -> list[dict[str, Any]]:
        infos = super()._process_extras(obs, terminated, truncated, extras, reset_ids)
        if not self.fast_variant:
            for idx in reset_ids:
                for key in self._TERMINAL_METRIC_KEYS:
                    if key in infos[idx]:
                        infos[idx][key] = self._to_host_scalar(infos[idx][key])
        return infos


__all__ = ["Sb3VecEnvWrapper", "process_sb3_cfg", "vecnormalize_path_for_checkpoint"]
