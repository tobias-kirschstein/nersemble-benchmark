import numpy as np
import torch
import trimesh
from dreifus.matrix import Pose
from flame_model import FlameConfig, FLAME

from nersemble_benchmark.data.benchmark_data import FlameTracking


class FlameProvider:

    def __init__(self, flame_tracking: FlameTracking):
        flame_config = FlameConfig(
            shape_params=300,
            expression_params=100,
            batch_size=1,
        )
        flame_model = FLAME(flame_config)

        self.flame_model = flame_model

        separate_transformation = True
        self._separate_transformation = separate_transformation
        self.shape_params = torch.tensor(flame_tracking.shape)  # [1, 300]
        assert self.shape_params.shape[
                   0] == 1, "There should only be a single set of shape params for the whole sequence"
        self.expr_params = torch.tensor(flame_tracking.expression)  # [T, 100]
        self.pose_params = torch.cat(
            [torch.tensor(np.zeros_like(flame_tracking.rotation)) if separate_transformation else torch.tensor(flame_tracking.rotation),
             torch.tensor(flame_tracking.jaw)], dim=-1)  # [T, 6]
        self.neck_pose = torch.tensor(flame_tracking.neck)
        self.eyes_pose = torch.tensor(flame_tracking.eyes)
        self.rotation = torch.tensor(flame_tracking.rotation)
        self.translation = torch.tensor(flame_tracking.translation)  # [T, 3]
        self.scale = torch.tensor(flame_tracking.scale)  # [1, 3]

        self._T = self.expr_params.shape[0]

    def get_vertices(self, timestep: int) -> np.array:
        i = timestep

        # FLAME forward
        flame_vertices, flame_lms = self.flame_model.forward(
            shape_params=self.shape_params[[0]],  # We always assume the same shape params for all timesteps
            expression_params=self.expr_params[[i]],
            pose_params=self.pose_params[[i]],
            neck_pose=None if self.neck_pose is None else self.neck_pose[[i]],
            eye_pose=None if self.eyes_pose is None else self.eyes_pose[[i]],
            transl=None if self._separate_transformation else self.translation[[i]])

        flame_vertices = self.apply_model_to_world_transformation(flame_vertices, timestep)
        flame_vertices = flame_vertices[0].detach().cpu().numpy()

        return flame_vertices

    def get_landmarks(self, timestep: int) -> torch.Tensor:
        i = timestep

        # FLAME forward
        _, flame_lms = self.flame_model.forward(
            shape_params=self.shape_params[[0]],  # We always assume the same shape params for all timesteps
            expression_params=self.expr_params[[i]],
            pose_params=self.pose_params[[i]],
            neck_pose=None if self.neck_pose is None else self.neck_pose[[i]],
            eye_pose=None if self.eyes_pose is None else self.eyes_pose[[i]],
            transl=None if self._separate_transformation else self.translation[[i]])

        flame_lms = self.apply_model_to_world_transformation(flame_lms, timestep)
        flame_lms = flame_lms[0].detach().cpu().numpy()

        return flame_lms

    def get_mesh(self, timestep: int) -> trimesh.Trimesh:
        vertices = self.get_vertices(timestep)
        mesh = self.create_mesh(vertices)
        return mesh

    def get_model_to_world_transformation(self, timestep: int) -> Pose:
        i = timestep
        model_to_world = Pose.from_euler(self.rotation[i].numpy(), self.translation[i].numpy(), 'XYZ')
        model_to_world[:3, :3] *= self.scale[0].item()
        return model_to_world

    def apply_model_to_world_transformation(self, points: torch.Tensor, timestep: int) -> torch.Tensor:
        i = timestep
        points_world = points
        if self._separate_transformation:
            B = points.shape[0]
            V = points.shape[1]
            model_transformations = torch.stack([torch.from_numpy(
                Pose.from_euler(self.rotation[i].numpy(), self.translation[i].numpy(), 'XYZ'))])
            model_transformations[:, :3, :3] *= self.scale[0]
            points_world = torch.cat([points_world, torch.ones((B, V, 1))], dim=-1)
            points_world = torch.bmm(points_world, model_transformations.permute(0, 2, 1))
            points_world = points_world[..., :3]

        return points_world

    def has_mesh(self, timestep: int) -> bool:
        return 0 <= timestep < self._T

    def create_mesh(self, vertices: np.ndarray) -> trimesh.Trimesh:
        flame_mesh = trimesh.Trimesh(vertices, self.flame_model.faces, process=False)

        return flame_mesh
