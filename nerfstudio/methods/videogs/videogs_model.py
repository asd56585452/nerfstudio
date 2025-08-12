"""
VideoGS model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Optional

import torch
from gsplat.rendering import rasterization
from torch import nn

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.cameras.cameras import Cameras

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


@dataclass
class VideoGSModelConfig(SplatfactoModelConfig):
    """VideoGS Model Config"""
    _target: Type = field(default_factory=lambda: VideoGSModel)
    temporal_lr: float = 1e-4
    """Learning rate for the temporal deformation network."""
    temporal_loss_weight: float = 0.1
    """Weight for the temporal regularization loss."""
    num_frames: int = 100
    """Number of frames in the video sequence."""

class TemporalDeformationNetwork(nn.Module):
    """A network that predicts deformation for Gaussians over time."""

    def __init__(self, num_frames: int):
        super().__init__()
        self.num_frames = num_frames

        # Using a similar TCNN architecture as in VideoGS's GlobalTField
        # but with time as an additional input.
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        }
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }

        # 4D input: x, y, z, time
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=4,
            n_output_dims=3, # Output is delta_xyz
            encoding_config=encoding_config,
            network_config=network_config,
        )

    def forward(self, xyz: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Canonical positions of Gaussians
            times: Time for each camera/ray
        """
        # Normalize time to [0, 1]
        normalized_times = times / self.num_frames

        # Ensure times is broadcastable to xyz
        if times.shape[0] != xyz.shape[0]:
            if times.numel() == 1:
                times = times.expand(xyz.shape[0], 1)
            else:
                # This case might happen if we have per-ray times
                # For now, we assume a single time value for the whole batch
                times = times[0].expand(xyz.shape[0], 1)

        # Concatenate position and time
        inputs = torch.cat([xyz, normalized_times], dim=-1)

        delta_xyz = self.model(inputs)
        return delta_xyz


class VideoGSModel(SplatfactoModel):
    """
    VideoGS model that extends Splatfacto to handle dynamic scenes.
    """
    config: VideoGSModelConfig

    def populate_modules(self):
        """Set up the fields for the model."""
        super().populate_modules()

        self.deformation_network = TemporalDeformationNetwork(num_frames=self.config.num_frames)

        # Store previous frame's gaussians for temporal loss
        self.prev_gaussians = nn.ParameterDict(
            {name: torch.zeros_like(param) for name, param in self.gauss_params.items()}
        )
        self.register_buffer("is_first_frame", torch.tensor(True), persistent=True)


    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["deformation_network"] = list(self.deformation_network.parameters())
        return param_groups

    def get_outputs(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """Takes in a camera and returns a dictionary of outputs."""

        if camera.times is None:
            # If no time is provided, default to the first frame.
            camera.times = torch.zeros(camera.shape[0], 1, dtype=torch.float32, device=self.device)

        # Get the canonical gaussian parameters
        canonical_means = self.means
        canonical_quats = self.quats
        canonical_scales = self.scales
        canonical_opacities = self.opacities
        canonical_features_dc = self.features_dc
        canonical_features_rest = self.features_rest

        # Predict deformation
        delta_xyz = self.deformation_network(canonical_means, camera.times)

        # Apply deformation
        deformed_means = canonical_means + delta_xyz

        # The rest of the rendering logic is copied from SplatfactoModel.get_outputs
        # but uses the deformed means.
        # ---- Start of copied logic ----
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        viewmat = get_viewmat(optimized_camera_to_world)

        # TODO: Handle cropping for dynamic scenes if needed
        crop_ids = None

        if crop_ids is not None:
            # This part needs to be adapted if cropping is used with dynamic gaussians
            raise NotImplementedError("Cropping for dynamic scenes not implemented yet.")
        else:
            opacities_crop = canonical_opacities
            means_crop = deformed_means # USE DEFORMED MEANS
            features_dc_crop = canonical_features_dc
            features_rest_crop = canonical_features_rest
            scales_crop = canonical_scales
            quats_crop = canonical_quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB",
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if hasattr(self.strategy, 'absgrad') else False,
            rasterize_mode=self.config.rasterize_mode,
        )

        alpha = alpha[:, ...]
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        depth_im = render[:, ..., 3:4] if (self.config.output_depth_during_training or not self.training) else None
        if depth_im is not None:
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)

        # ---- End of copied logic ----

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Temporal loss
        if self.training and not self.is_first_frame:
            temporal_loss = 0.0

            # L2 loss on the difference in gaussian attributes between frames
            # Note: for simplicity, we only regularize the canonical parameters
            # A more advanced version might regularize the predicted deformations.
            for name, param in self.gauss_params.items():
                if name in self.prev_gaussians:
                    temporal_loss += torch.mean((param - self.prev_gaussians[name].detach()) ** 2)

            loss_dict["temporal_loss"] = self.config.temporal_loss_weight * temporal_loss
            loss_dict["main_loss"] += loss_dict["temporal_loss"]

        return loss_dict

    def step_cb(self, optimizers: 'Optimizers', step):
        super().step_cb(optimizers, step)

        # After each step, update the previous frame's gaussians
        # This is a simplification. In a real scenario, we'd do this at frame boundaries.
        # For now, we do it every step to simulate the temporal regularization.
        if self.training:
            if self.is_first_frame:
                self.is_first_frame = False

            with torch.no_grad():
                for name, param in self.gauss_params.items():
                    if name in self.prev_gaussians:
                        self.prev_gaussians[name].copy_(param.detach())
