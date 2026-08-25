"""Local depth-supervised Splatfacto method for Arena RGB-D caches.

The module is discovered by Nerfstudio through ``NERFSTUDIO_METHOD_CONFIGS``.
It intentionally extends, rather than patches, the installed Nerfstudio package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Type

import torch
import torch.nn.functional as functional
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, resize_image
from nerfstudio.plugins.types import MethodSpecification


class DepthFullImageDatamanager(FullImageDatamanager[DepthDataset]):
    """Full-image datamanager that retains and transfers metric depth batches."""

    def next_train(self, step: int):
        camera, batch = super().next_train(step)
        if "depth_image" in batch:
            batch["depth_image"] = batch["depth_image"].to(self.device)
        return camera, batch

    def next_eval_image(self, step: int):
        camera, batch = super().next_eval_image(step)
        if "depth_image" in batch:
            batch["depth_image"] = batch["depth_image"].to(self.device)
        return camera, batch


@dataclass
class DepthFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    """Configuration for the depth-aware full-image datamanager."""

    _target: Type = field(default_factory=lambda: DepthFullImageDatamanager)


@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto configuration with guarded metric-depth supervision."""

    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    output_depth_during_training: bool = True
    depth_loss_weight: float = 0.05
    depth_warmup_steps: int = 500
    depth_ramp_steps: int = 1500
    depth_huber_delta_m: float = 0.02
    depth_minimum_m: float = 0.10
    depth_maximum_m: float = 3.0
    depth_edge_threshold_m: float = 0.06
    depth_minimum_opacity: float = 0.05


class DepthSplatfactoModel(SplatfactoModel):
    """Splatfacto with a conservative loss against sensor metric depth."""

    config: DepthSplatfactoModelConfig

    def _depth_loss_weight(self) -> float:
        if self.step < self.config.depth_warmup_steps:
            return 0.0
        if self.config.depth_ramp_steps <= 0:
            return self.config.depth_loss_weight
        progress = min(
            1.0,
            (self.step - self.config.depth_warmup_steps + 1)
            / self.config.depth_ramp_steps,
        )
        return self.config.depth_loss_weight * progress

    def _valid_depth_mask(
        self, target_depth: torch.Tensor, accumulation: torch.Tensor
    ) -> torch.Tensor:
        """Reject invalid depth, discontinuities, and unsupported render regions."""
        valid = torch.isfinite(target_depth)
        valid &= target_depth >= self.config.depth_minimum_m
        valid &= target_depth <= self.config.depth_maximum_m

        # A 3x3 max-min test removes depth silhouettes and invalid-depth borders,
        # where a metric surface loss would otherwise pull splats across edges.
        depth_4d = target_depth[None, None, ...]
        local_max = functional.max_pool2d(depth_4d, kernel_size=3, stride=1, padding=1)
        local_min = -functional.max_pool2d(-depth_4d, kernel_size=3, stride=1, padding=1)
        valid &= (local_max - local_min)[0, 0] <= self.config.depth_edge_threshold_m
        valid &= accumulation.squeeze(-1) >= self.config.depth_minimum_opacity
        return valid

    def get_loss_dict(
        self, outputs: dict[str, Any], batch: dict[str, Any], metrics_dict=None
    ) -> dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        zero = outputs["rgb"].new_zeros(())
        weight = self._depth_loss_weight()
        target_depth = batch.get("depth_image")
        rendered_depth = outputs.get("depth")

        if weight <= 0.0 or target_depth is None or rendered_depth is None:
            loss_dict["depth_loss"] = zero
            if metrics_dict is not None:
                metrics_dict["depth_valid_fraction"] = zero
                metrics_dict["depth_loss_weight"] = zero.new_tensor(weight)
            return loss_dict

        target_depth = self._downscale_if_required(target_depth).to(
            device=self.device, dtype=rendered_depth.dtype
        )
        # Nerfstudio 1.1.5 returns expected rendered depth as HxWx1.  Accept
        # either that representation or HxW, then use one canonical HxW form.
        if rendered_depth.ndim == 3 and rendered_depth.shape[-1] == 1:
            rendered_depth = rendered_depth[..., 0]
        if target_depth.ndim == 3 and target_depth.shape[-1] == 1:
            target_depth = target_depth[..., 0]
        if target_depth.ndim != 2 or rendered_depth.ndim != 2:
            raise ValueError(
                "Rendered and sensor depth must be HxW or HxWx1, got "
                f"{tuple(rendered_depth.shape)} and {tuple(target_depth.shape)}"
            )
        if target_depth.shape != rendered_depth.shape:
            raise ValueError(
                "Rendered depth and sensor depth shapes differ: "
                f"{tuple(rendered_depth.shape)} vs {tuple(target_depth.shape)}"
            )

        valid = self._valid_depth_mask(target_depth, outputs["accumulation"])
        valid_count = valid.sum()
        if valid_count.item() == 0:
            raw_depth_loss = zero
        else:
            raw_depth_loss = functional.smooth_l1_loss(
                rendered_depth[valid],
                target_depth[valid],
                beta=self.config.depth_huber_delta_m,
                reduction="mean",
            )
        loss_dict["depth_loss"] = raw_depth_loss * weight
        if metrics_dict is not None:
            metrics_dict["depth_loss_unscaled"] = raw_depth_loss.detach()
            metrics_dict["depth_valid_fraction"] = valid.float().mean().detach()
            metrics_dict["depth_loss_weight"] = zero.new_tensor(weight)
        return loss_dict


_depth_splatfacto_config = deepcopy(method_configs["splatfacto"])
_depth_splatfacto_config.method_name = "depth-splatfacto"
_depth_splatfacto_config.pipeline.datamanager = DepthFullImageDatamanagerConfig(
    dataparser=_depth_splatfacto_config.pipeline.datamanager.dataparser,
    cache_images_type="uint8",
)
_depth_splatfacto_config.pipeline.model = DepthSplatfactoModelConfig()

depth_splatfacto_method = MethodSpecification(
    config=_depth_splatfacto_config,
    description="Splatfacto with conservative metric RGB-D depth supervision.",
)
