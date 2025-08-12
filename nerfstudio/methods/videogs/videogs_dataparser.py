"""
Data parser for VideoGS datasets.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

@dataclass
class VideoGSDataParserConfig(DataParserConfig):
    """VideoGS dataset parser config."""

    _target: Type = field(default_factory=lambda: VideoGSDataParser)
    """target class to instantiate"""
    data: Path = Path("data/dycheck/mochi-high-five")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""


@dataclass
class VideoGSDataParser(DataParser):
    """VideoGS Dataparser
    Assumes the data is stored in the VideoGS format, where each frame has its own
    directory with a transforms.json file.

    For example:
    data/
        0/
            images/
                0.png
                1.png
                ...
            transforms.json
        1/
            images/
                0.png
                1.png
                ...
            transforms.json
        ...
    """

    config: VideoGSDataParserConfig

    def __init__(self, config: VideoGSDataParserConfig):
        super().__init__(config=config)
        self.data = config.data
        self.scale_factor = config.scale_factor
        self.scene_scale = config.scene_scale

    def _generate_dataparser_outputs(self, split="train"):
        image_filenames = []
        poses = []
        camera_times = []
        fxs = []
        fys = []
        cxs = []
        cys = []

        # This logic is adapted from `readCamerasFromTransforms` in VideoGS
        def process_transforms(frame_dir: Path, frame_idx: int):
            meta = load_from_json(frame_dir / "transforms.json")
            frames = meta["frames"]
            for frame in frames:
                # Image filename
                fname = frame_dir / frame["file_path"]
                image_filenames.append(fname)

                # Append camera time
                camera_times.append(frame_idx)

                # Intrinsics
                fxs.append(frame["fl_x"])
                fys.append(frame["fl_y"])
                cxs.append(frame["cx"])
                cys.append(frame["cy"])

                # Extrinsics
                # VideoGS uses a different coordinate system convention
                flip_mat = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)

                # Convert from OpenCV to OpenGL coordinate system
                c2w = np.linalg.inv(np.matmul(transform_matrix, flip_mat))

                # The VideoGS camera coordinate system is rotated 180 degrees around the x-axis
                # compared to the NeRF (and Nerfstudio) coordinate systems.
                # We can correct for this by rotating the camera-to-world matrix.
                c2w[0:3, 1:3] *= -1

                poses.append(c2w)


        frame_dirs = sorted([d for d in self.data.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
        for frame_dir in frame_dirs:
            frame_idx = int(frame_dir.name)
            process_transforms(frame_dir, frame_idx)

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        camera_times = torch.tensor(camera_times, dtype=torch.float32)

        # Scale poses
        poses[:, :3, 3] *= self.scale_factor

        # Create cameras
        fx = torch.tensor(fxs, dtype=torch.float32)
        fy = torch.tensor(fys, dtype=torch.float32)
        cx = torch.tensor(cxs, dtype=torch.float32)
        cy = torch.tensor(cys, dtype=torch.float32)

        # Assuming all images have the same dimensions, read from the first one
        img_0 = Image.open(image_filenames[0])
        w, h = img_0.size

        # Use camera_to_worlds from poses
        camera_to_worlds = poses[:, :3, :4]

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            camera_type=CameraType.PERSPECTIVE,
            times=camera_times,
        )

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.scene_scale, -self.scene_scale, -self.scene_scale], [self.scene_scale, self.scene_scale, self.scene_scale]], dtype=torch.float32
            )
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
