import os
import cv2
import json
import torch
import smplx
import shutil
import base64
import numpy as np
import pandas as pd
import open3d as o3d
import altair as alt
import pickle as pkl
import pytorch_lightning as pl
from PIL import Image
from io import BytesIO
from typing import Any, List, Optional, Tuple
from torch import nn
from pathlib import Path
from flame import FLAME
from scipy.spatial.transform import Rotation
from typing import Tuple, Literal, List, Dict, Any, Optional, Union
from torch.nn import functional as F
from attrdict import AttrDict
from pytorch_lightning import Callback
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
    BlendParams,
    Materials,
    AmbientLights,
)
from smal_layer import get_smal_layer


class C2M(nn.Module):
    def __init__(self, num_stats: int, hidden_size: int, out_features: int, num_hiddens: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(num_stats, hidden_size)
        self.fc5 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc5(x)
        return x


class C2M_new(nn.Module):
    def __init__(self, num_stats: int, hidden_size: int, out_features: int, num_hiddens: int = 0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_stats, hidden_size))
        if num_hiddens > 0:
            for _ in range(num_hiddens):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.out_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        return self.out_layer(x)


class Open3dRenderer:
    def __init__(
        self,
        verts: torch.tensor,
        faces: torch.tensor,
        vt: torch.tensor = None,
        ft: torch.tensor = None,
        texture: str = None,
        light_on: bool = True,
        for_image: bool = True,
        img_size: Tuple[int, int] = (512, 512),
        paint_vertex_colors: bool = False,
    ):
        self.verts = verts
        self.faces = faces
        self.height, self.width = img_size
        self.paint_vertex_colors = paint_vertex_colors
        self.texture = cv2.cvtColor(cv2.imread(texture), cv2.COLOR_BGR2RGB) if texture is not None else None
        self.vt = vt
        self.ft = ft
        if self.vt is not None and self.ft is not None:
            uvs = np.concatenate([self.vt[self.ft[:, ind]][:, None] for ind in range(3)], 1).reshape(-1, 2)
            uvs[:, 1] = 1 - uvs[:, 1]
        else:
            uvs = None
        self.uvs = uvs
        self.for_image = for_image
        self.visualizer = o3d.visualization.Visualizer()
        self.default_zoom_value = 0.55
        self.default_y_rotate_value = 70.0
        self.default_up_translate_value = 0.3
        self.visualizer.create_window(width=self.width, height=self.height)
        opt = self.visualizer.get_render_option()
        if self.paint_vertex_colors:
            opt.background_color = np.asarray([255.0, 255.0, 255.0])
        else:
            opt.background_color = np.asarray([0.0, 0.0, 0.0])
        self.visualizer.get_render_option().light_on = light_on
        self.ctr = self.visualizer.get_view_control()
        self.ctr.set_zoom(self.default_zoom_value)
        self.ctr.camera_local_rotate(0.0, self.default_y_rotate_value, 0.0)
        self.ctr.camera_local_translate(0.0, 0.0, self.default_up_translate_value)
        self.mesh = self.get_initial_mesh()
        self.visualizer.add_geometry(self.mesh)
        self.mesh.compute_vertex_normals()

    def get_texture(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:

        if self.texture is not None and isinstance(self.texture, str):
            self.texture = cv2.cvtColor(cv2.imread(self.texture), cv2.COLOR_BGR2RGB)
        mesh.textures = [o3d.geometry.Image(self.texture)]
        mesh.triangle_uvs = o3d.utility.Vector2dVector(self.uvs)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def get_initial_mesh(self) -> o3d.geometry.TriangleMesh:
        verts = (self.verts.squeeze() - self.verts.min()) / (self.verts.max() - self.verts.min())
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.squeeze())
        if self.texture is not None:
            mesh = self.get_texture(mesh)

        if self.uvs is not None:
            mesh.triangle_uvs = o3d.utility.Vector2dVector(self.uvs)

        if self.paint_vertex_colors:
            mesh.paint_uniform_color([0.2, 0.8, 0.2])

        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(self.faces)), dtype=np.int32))
        return mesh

    def render_mesh(self, verts: torch.tensor = None, texture: np.array = None):

        if verts is not None:
            verts = (verts.squeeze() - verts.min()) / (verts.max() - verts.min())
            self.mesh.vertices = o3d.utility.Vector3dVector(verts.squeeze())
            self.visualizer.update_geometry(self.mesh)
        if texture is not None:
            self.texture = texture
            self.mesh = self.get_texture(self.mesh)
            self.visualizer.update_geometry(self.mesh)
        if self.for_image:
            self.visualizer.update_renderer()
            self.visualizer.poll_events()
        else:
            self.visualizer.run()

    def close(self):
        self.visualizer.close()

    def remove_texture(self):
        self.mesh.textures = []
        self.visualizer.update_geometry(self.mesh)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def zoom(self, zoom_value: float):
        self.ctr.set_zoom(zoom_value)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def rotate(self, y: float = None, x: float = 0.0, z: float = 0.0):
        if y is None:
            y = self.default_y_rotate_value
        self.ctr.camera_local_rotate(x, y, z)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def translate(self, right: float = 0.0, up: float = None):
        if up is None:
            up = self.default_up_translate_value
        self.ctr.camera_local_translate(0.0, right, up)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def reset_camera(self):
        self.ctr.set_zoom(self.default_zoom_value)
        self.visualizer.update_renderer()
        self.visualizer.poll_events()

    def save_mesh(self, path: str):
        o3d.io.write_triangle_mesh(path, self.mesh)


class SMPLXParams:
    def __init__(self, betas: torch.tensor = None, expression: torch.tensor = None, body_pose: torch.tensor = None):
        if betas is not None:
            betas: torch.Tensor = betas
        else:
            betas = torch.zeros(1, 10)
        if expression is not None:
            expression: torch.Tensor = expression
        else:
            expression: torch.Tensor = torch.zeros(1, 10)
        if body_pose is not None:
            body_pose: torch.Tensor = body_pose
        else:
            body_pose = torch.eye(3).expand(1, 21, 3, 3)
        left_hand_pose: torch.Tensor = torch.eye(3).expand(1, 15, 3, 3)
        right_hand_pose: torch.Tensor = torch.eye(3).expand(1, 15, 3, 3)
        global_orient: torch.Tensor = torch.eye(3).expand(1, 1, 3, 3)
        transl: torch.Tensor = torch.zeros(1, 3)
        jaw_pose: torch.Tensor = torch.eye(3).expand(1, 1, 3, 3)
        self.params = {
            "betas": betas,
            "body_pose": body_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "global_orient": global_orient,
            "transl": transl,
            "jaw_pose": jaw_pose,
            "expression": expression,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class FLAMEParams:
    def __init__(
        self,
        shape_params: torch.tensor = None,
        expression_params: torch.tensor = None,
        jaw_pose: float = None,
    ):
        if shape_params is None:
            shape_params = torch.zeros(1, 100, dtype=torch.float32)
        if expression_params is None:
            expression_params = torch.zeros(1, 50, dtype=torch.float32)
        shape_params = shape_params.cuda()
        expression_params = expression_params.cuda()
        if jaw_pose is None:
            pose_params_t = torch.zeros(1, 6, dtype=torch.float32)
        else:
            pose_params_t = torch.cat([torch.zeros(1, 3), torch.tensor([[jaw_pose, 0.0, 0.0]])], 1)
        pose_params = pose_params_t.cuda()
        self.params = {
            "shape_params": shape_params,
            "expression_params": expression_params,
            "pose_params": pose_params,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class SMALParams:
    def __init__(self, beta: torch.tensor = None):
        if beta is None:
            beta = torch.zeros(1, 41, dtype=torch.float32)
        beta = beta.cuda()
        theta = torch.eye(3).expand(1, 35, 3, 3).to("cuda")
        self.params = {
            "beta": beta,
            "theta": theta,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class Pytorch3dRenderer:
    def __init__(
        self,
        device="cuda",
        dist: float = 0.5,
        elev: float = 0.0,
        azim: float = 0.0,
        img_size: Tuple[int, int] = (224, 224),
        tex_path: str = None,
    ):

        self.device = device
        self.tex_map = cv2.cvtColor(cv2.imread(tex_path), cv2.COLOR_BGR2RGB) if tex_path is not None else None
        self.height, self.width = img_size

        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev)
        self.cameras = FoVPerspectiveCameras(znear=0.1, T=T, R=R, fov=30).to(self.device)
        lights = self.get_lights(self.device)
        materials = self.get_default_materials(self.device)
        blend_params = self.get_default_blend_params()
        self.shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        )

    @staticmethod
    def get_texture(device, vt, ft, texture):
        verts_uvs = torch.as_tensor(vt, dtype=torch.float32, device=device)
        faces_uvs = torch.as_tensor(ft, dtype=torch.long, device=device)

        texture_map = torch.as_tensor(texture, device=device, dtype=torch.float32) / 255.0

        texture = TexturesUV(
            maps=texture_map[None],
            faces_uvs=faces_uvs[None],
            verts_uvs=verts_uvs[None],
        )
        return texture

    @staticmethod
    def get_lights(device):
        lights = PointLights(
            device=device,
            ambient_color=((0.8, 0.8, 0.8),),
            specular_color=((0.2, 0.2, 0.2),),
            location=[[0.0, 2.0, 2.0]],
        )
        return lights

    @staticmethod
    def get_default_materials(device):
        materials = Materials(device=device)  # , shininess=12)
        return materials

    @staticmethod
    def get_default_blend_params():
        blend_params = BlendParams(sigma=1e-6, gamma=1e-6, background_color=(255.0, 255.0, 255.0))
        return blend_params

    def render_mesh(self, verts, faces, vt=None, ft=None):
        verts = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        if self.tex_map is not None:
            assert vt is not None and ft is not None, "vt and ft must be provided if texture is provided"
            texture = self.get_texture(self.device, vt, ft, self.tex_map)
        else:
            if len(verts.shape) == 2:
                verts = verts[None]

            texture = TexturesVertex(
                verts_features=torch.ones(*verts.shape, device=self.device)
                * torch.tensor([0.7, 0.7, 0.7], device=self.device)
            )
        if len(verts.size()) == 2:
            verts = verts[None]
        if len(faces.size()) == 2:
            faces = faces[None]
        mesh = Meshes(verts=verts, faces=faces, textures=texture)
        raster_settings = RasterizationSettings(image_size=(self.height, self.width))
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=self.shader,
        )
        rendered_mesh = renderer(mesh, cameras=self.cameras)
        return rendered_mesh

    @staticmethod
    def save_rendered_image(image, path):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().squeeze()
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class Utils:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.body_pose = torch.tensor(np.load("/home/nadav2/dev/repos/CLIP2Shape/SMPLX/rest_pose.npy"))
        self.production_dir = "/home/nadav2/dev/repos/CLIP2Shape/pre_production"

    @staticmethod
    def find_multipliers(value: int) -> list:
        """
        Description
        -----------
        finds all of the pairs that their product is the value
        Args
        ----
        value (int) = a number that you would like to get its multipliers
        Returns
        -------
        list of the pairs that their product is the value
        """
        factors = []
        for i in range(1, int(value**0.5) + 1):
            if value % i == 0:
                factors.append((i, value / i))
        return factors

    def get_plot_shape(self, value: int) -> Tuple[Tuple[int, int], int]:
        """
        Description
        -----------
        given a number it finds the best pair of integers that their product
        equals the given number.
        for example, given an input 41 it will return 5 and 8
        """
        options_list = self.find_multipliers(value)
        if len(options_list) == 1:
            while len(options_list) == 1:
                value -= 1
                options_list = self.find_multipliers(value)

        chosen_multipliers = None
        min_distance = 100
        for option in options_list:
            if abs(option[0] - option[1]) < min_distance:
                chosen_multipliers = (option[0], option[1])

        # it is better that the height will be the largest value since the image is wide
        chosen_multipliers = (
            int(chosen_multipliers[np.argmax(chosen_multipliers)]),
            int(chosen_multipliers[1 - np.argmax(chosen_multipliers)]),
        )

        return chosen_multipliers, int(value)

    @staticmethod
    def flatten_list_of_lists(list_of_lists):
        return [l[0] for l in list_of_lists]

    @staticmethod
    def create_metadata(metadata: Dict[str, torch.tensor], file_path: str):
        # write tensors to json
        for key, value in metadata.items():
            metadata[key] = value.tolist()

        with open(file_path, "w") as f:
            json.dump(metadata, f)

    def _get_smplx_layer(self, gender: str):
        if gender == "neutral":
            smplx_path = "/home/nadav2/dev/repos/CLIP2Shape/SMPLX/SMPLX_NEUTRAL_2020.npz"
        elif gender == "male":
            smplx_path = "/home/nadav2/dev/repos/CLIP2Shape/SMPLX/SMPLX_MALE.npz"
        else:
            smplx_path = "/home/nadav2/dev/repos/CLIP2Shape/SMPLX/SMPLX_FEMALE.npz"
        self.smplx_layer = smplx.build_layer(model_path=smplx_path, num_expression_coeffs=10)
        model_data = np.load(smplx_path, allow_pickle=True)
        self.smplx_faces = model_data["f"].astype(np.int32)

    def _get_flame_layer(self, gender: Literal["male", "female", "neutral"]) -> FLAME:
        cfg = self.get_flame_model_kwargs(gender)
        self.flame_layer = FLAME(cfg).cuda()

    def get_smplx_model(
        self,
        betas: torch.tensor = None,
        body_pose: torch.tensor = None,
        expression: torch.tensor = None,
        gender: Literal["neutral", "male", "female"] = "neutral",
        device: Optional[Literal["cuda", "cpu"]] = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        smplx_model = SMPLXParams(betas=betas, body_pose=body_pose, expression=expression)
        if not hasattr(self, "smplx_layer") or not hasattr(self, "smplx_faces"):
            self._get_smplx_layer(gender)

        if device == "cuda":
            smplx_model.params = smplx_model.to(device)
            self.smplx_layer = self.smplx_layer.cuda()
            verts = self.smplx_layer(**smplx_model.params).vertices
        else:
            verts = self.smplx_layer(**smplx_model.params).vertices
            verts = verts.detach().cpu().numpy()
        verts = (verts.squeeze() - verts.min()) / (verts.max() - verts.min())
        verts = self.translate_mesh_smplx(verts)
        if not hasattr(self, "vt") and not hasattr(self, "ft"):
            self._get_vt_ft("smplx")

        return verts, self.smplx_faces, self.vt, self.ft

    def _get_vt_ft(self, model_type: Literal["smplx", "flame"]) -> Tuple[np.ndarray, np.ndarray]:
        if model_type == "smplx":
            vt = np.load("/home/nadav2/dev/repos/CLIP2Shape/SMPLX/textures/smplx_vt.npy")
            ft = np.load("/home/nadav2/dev/repos/CLIP2Shape/SMPLX/textures/smplx_ft.npy")
        else:
            flame_uv_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/flame_texture_data_v6.pkl"
            flame_uv = np.load(flame_uv_path, allow_pickle=True)
            vt = flame_uv["vt_plus"]
            ft = flame_uv["ft_plus"]
        self.vt, self.ft = vt, ft
        return vt, ft

    def _get_flame_faces(self) -> np.ndarray:
        flame_uv_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/flame_texture_data_v6.pkl"
        flame_uv = np.load(flame_uv_path, allow_pickle=True)
        self.flame_faces = flame_uv["f_plus"]

    def _get_smal_faces(self) -> np.ndarray:
        smal_model_path = "/home/nadav2/dev/repos/CLIP2Shape/SMAL/smal_CVPR2017.pkl"
        with open(smal_model_path, "rb") as f:
            smal_model = pkl.load(f, encoding="latin1")
        self.smal_faces = smal_model["f"].astype(np.int32)

    @staticmethod
    def init_flame_params_dict(device: str = "cuda") -> Dict[str, torch.tensor]:
        flame_dict = {}
        flame_dict["shape_params"] = torch.zeros(1, 300)
        flame_dict["expression_params"] = torch.zeros(1, 100)
        flame_dict["global_rot"] = torch.zeros(1, 3)
        flame_dict["jaw_pose"] = torch.zeros(1, 3)
        flame_dict["neck_pose"] = torch.zeros(1, 3)
        flame_dict["transl"] = torch.zeros(1, 3)
        flame_dict["eye_pose"] = torch.zeros(1, 6)
        flame_dict["shape_offsets"] = torch.zeros(1, 5023, 3)
        flame_dict = {k: v.to(device) for k, v in flame_dict.items()}
        return flame_dict

    @staticmethod
    def get_flame_model_kwargs(gender: Literal["male", "female", "neutral"]) -> Dict[str, Any]:
        if gender == "male":
            flame_model_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/male_model.pkl"
        elif gender == "female":
            flame_model_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame/female_model.pkl"
        else:
            flame_model_path = "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/generic_model.pkl"

        kwargs = {
            "batch_size": 1,
            "use_face_contour": False,
            "use_3D_translation": True,
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "shape_params": 100,
            "expression_params": 50,
            "flame_model_path": flame_model_path,
            "ring_margin": 0.5,
            "ring_loss_weight": 1.0,
            "static_landmark_embedding_path": "/home/nadav2/dev/repos/CLIP2Shape/Flame/flame2020/flame_static_embedding_68.pkl",
            "pose_params": 6,
        }
        return AttrDict(kwargs)

    def get_flame_model(
        self,
        shape_params: torch.tensor = None,
        expression_params: torch.tensor = None,
        jaw_pose: float = None,
        gender: Literal["male", "female", "neutral"] = "neutral",
        device: Optional[Literal["cuda", "cpu"]] = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "flame_layer"):
            self._get_flame_layer(gender)
        if shape_params is not None and shape_params.shape == (1, 10):
            shape_params = torch.cat([shape_params, torch.zeros(1, 90).to(device)], dim=1)
        if expression_params is not None and expression_params.shape == (1, 10):
            expression_params = torch.cat([expression_params, torch.zeros(1, 40).to(device)], dim=1)
        flame_params = FLAMEParams(shape_params=shape_params, expression_params=expression_params, jaw_pose=jaw_pose)
        if device == "cuda":
            flame_params.params = flame_params.to(device)
        verts, _ = self.flame_layer(**flame_params.params)
        if device == "cpu":
            verts = verts.cpu()
        if not hasattr(self, "flame_faces"):
            self._get_flame_faces()

        if not hasattr(self, "vt") and not hasattr(self, "ft"):
            self._get_vt_ft("flame")

        return verts, self.flame_faces, self.vt, self.ft

    def get_smal_model(
        self, beta: torch.tensor, device: Optional[Literal["cuda", "cpu"]] = "cpu", py3d: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, None, None]:
        if not hasattr(self, "smal_layer"):
            self.smal_layer = get_smal_layer()
        smal_params = SMALParams(beta=beta)
        if device == "cuda":
            smal_params.params = smal_params.to(device)
            self.smal_layer = self.smal_layer.cuda()
            verts = self.smal_layer(**smal_params.params)[0]
        else:
            verts = self.smal_layer(**smal_params.params)[0].detach().cpu().numpy()
        verts = self.rotate_mesh_smal(verts, py3d)
        if not hasattr(self, "smal_faces"):
            self._get_smal_faces()
        return verts, self.smal_faces, None, None

    def get_body_pose(self) -> torch.tensor:
        return self.body_pose

    @staticmethod
    def rotate_mesh_smal(verts: Union[np.ndarray, torch.Tensor], py3d: bool = True) -> np.ndarray:
        rotation_matrix_x = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        rotation_matrix_y = Rotation.from_euler("y", 75, degrees=True).as_matrix()
        rotation_matrix = np.matmul(rotation_matrix_x, rotation_matrix_y)
        if not py3d:
            rotation_matrix_z = Rotation.from_euler("x", -15, degrees=True).as_matrix()
            rotation_matrix = np.matmul(rotation_matrix, rotation_matrix_z)
        mesh_center = verts.mean(axis=1)
        if isinstance(verts, torch.Tensor):
            mesh_center = torch.tensor(mesh_center).to(verts.device).float()
            rotation_matrix = torch.tensor(rotation_matrix).to(verts.device).float()
        verts = verts - mesh_center
        verts = verts @ rotation_matrix
        verts = verts + mesh_center
        return verts

    @staticmethod
    def translate_mesh_smplx(
        verts: Union[np.ndarray, torch.tensor],
        translation_vector: Union[np.ndarray, torch.tensor] = np.array([-0.55, -0.3, 0.0]),
    ) -> Union[np.ndarray, torch.tensor]:
        if isinstance(verts, torch.Tensor):
            translation_vector = torch.tensor(translation_vector).to(verts.device)
        verts += translation_vector
        return verts

    @staticmethod
    def get_labels() -> List[List[str]]:
        # labels = [["big cat"], ["cow"], ["donkey"], ["hippo"]]  # SMAL animals
        labels = [
            ["fat"],
            ["long legs"],
            ["curvy"],
            ["skinny arms"],
            ["pear shaped"],
            ["muscular"],
            ["big head"],
            # ["good fit"],
            ["long neck"],
        ]  # SMPLX body
        # labels = [
        #     ["smile"],
        #     ["angry"],
        #     ["sealed lips"],
        #     ["lifted eyebrows"],
        #     ["opened eyes"],
        # ]  # FLAME expression
        # labels = [
        #     ["fat"],
        #     ["long neck"],
        #     ["chubby cheeks"],
        #     ["nose sticking-out"],
        #     ["ears sticking-out"],
        #     ["big forehead"],
        #     ["small chin"],
        # ]  # FLAME shape
        return labels

    def get_antonyms_of_labels(self, labels: List[List[str]]):
        syn_ant_dict = self.syntonyms_antonyms()
        for sublist in labels:
            splitted_sublist = sublist[0].split(" ")
            if len(splitted_sublist) > 1:
                for word in splitted_sublist:
                    if word in syn_ant_dict.keys():
                        antonyms = syn_ant_dict[word]
                        replace_idx = splitted_sublist.index(word)
                        new_word_list = splitted_sublist.copy()
                        new_word_list[replace_idx] = antonyms
                        break
                new_word = [" ".join(new_word_list)]
                sublist += new_word
        return labels

    @staticmethod
    def syntonyms_antonyms():
        return {"open": "close", "raise": "drop", "narrow": "wide", "long": "short", "big": "small", "fat": "thin"}

    def get_num_stats(self) -> int:
        return len(self.get_labels())

    @staticmethod
    def get_random_betas_smplx() -> torch.tensor:
        """SMPLX body shape"""
        return torch.randn(1, 10) * torch.randint(-4, 4, (1, 10)).float()

    @staticmethod
    def get_random_betas_smal() -> torch.tensor:
        """SMAL body shape"""
        shape = torch.randn(1, 10)  # * torch.randint(-1, 1, (1, 10)).float()
        return torch.cat([shape, torch.zeros(1, 31)], dim=1)

    @staticmethod
    def get_random_expression() -> torch.tensor:
        """SMPLX face expression"""
        return torch.randn(1, 10) * torch.randint(-3, 3, (1, 10)).float()

    @staticmethod
    def get_random_shape() -> torch.tensor:
        """FLAME face shape"""
        shape = torch.rand(1, 10) * torch.randint(-3, 6, (1, 10)).float()
        return torch.cat([shape, torch.zeros(1, 90)], dim=1)

    @staticmethod
    def get_random_expression_flame() -> torch.tensor:
        """FLAME face expression"""
        expression = torch.randn(1, 10) * torch.randint(-3, 4, (1, 10)).float()
        return torch.cat([expression, torch.zeros(1, 40)], dim=1)

    @staticmethod
    def convert_str_list_to_float_tensor(strs_list: List[str]) -> torch.tensor:
        stats = [float(stat) for stat in strs_list[0].split(" ")]
        return torch.tensor(stats, dtype=torch.float32)[None]

    @staticmethod
    def normalize_data(data, min_max_dict):
        for key, value in data.items():
            min_val, max_val, _ = min_max_dict[key]
            data[key] = (value - min_val) / (max_val - min_val)
        return data

    @staticmethod
    def filter_params_hack(ckpt: Dict) -> Dict:
        hack = {key.split("model.")[-1]: ckpt["state_dict"][key] for key in ckpt["state_dict"].keys() if "model" in key}
        return hack

    def get_model_to_eval(self, model_path: str) -> nn.Module:
        model_meta_path = model_path.replace(".ckpt", "_metadata.json")
        with open(model_meta_path, "r") as f:
            model_meta = json.load(f)

        # kwargs from metadata
        labels = model_meta["labels"]
        model_meta.pop("labels")
        if "lr" in model_meta:
            model_meta.pop("lr")

        # load model
        ckpt = torch.load(model_path)
        filtered_params_hack = self.filter_params_hack(ckpt)
        model = C2M(**model_meta).to(self.device)
        model.load_state_dict(filtered_params_hack)
        model.eval()

        return model, labels

    @staticmethod
    def get_default_parameters(body_pose: bool = False) -> torch.tensor:
        if body_pose:
            return torch.eye(3).expand(1, 21, 3, 3)
        return torch.zeros(1, 10)

    @staticmethod
    def get_default_face_shape() -> torch.tensor:
        return torch.zeros(1, 100)

    @staticmethod
    def get_default_face_expression() -> torch.tensor:
        return torch.zeros(1, 50)

    @staticmethod
    def get_default_shape_smal() -> torch.tensor:
        return torch.zeros(1, 41)

    @staticmethod
    def get_min_max_values(working_dir: str) -> Dict[str, Tuple[float, float, float]]:
        stats = {}
        min_max_dict = {}

        for file in Path(working_dir).rglob("*_labels.json"):
            with open(file.as_posix(), "r") as f:
                data = json.load(f)
            for key, value in data.items():
                if key not in stats:
                    stats[key] = []
                stats[key].append(value)

        for key, value in stats.items():
            stats[key] = np.array(value)
            # show min and max
            min_max_dict[key] = (np.min(stats[key]), np.max(stats[key]), np.mean(stats[key]))
        return min_max_dict


class C2M_pl(pl.LightningModule):
    def __init__(
        self, num_stats: int, lr: float = 0.0001, out_features: int = 10, hidden_size: int = 300, num_hiddens: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = C2M(
            num_stats=num_stats, out_features=out_features, hidden_size=hidden_size, num_hiddens=num_hiddens
        )
        self.lr = lr
        self.utils = Utils()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        parameters, clip_labels = batch
        b = parameters.shape[0]
        parameters_pred = self(clip_labels)
        parameters_pred = parameters_pred.reshape(b, 1, 10)
        loss = F.mse_loss(parameters, parameters_pred)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class CreateModelMeta(Callback):
    def __init__(self):
        self.utils = Utils()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path is None:
            return
        ckpt_new_name = f"{trainer.logger.name}.ckpt"
        ckpt_new_path = ckpt_path.replace(ckpt_path.split("/")[-1], ckpt_new_name)
        os.rename(ckpt_path, ckpt_new_path)
        shutil.copy(ckpt_new_path, f"{self.utils.production_dir}/{ckpt_new_name}")
        metadata = {"labels": self.utils.get_labels()}
        metadata.update(pl_module.hparams)
        with open(
            f"{self.utils.production_dir}/{ckpt_new_path.split('/')[-1].replace('.ckpt', '_metadata.json')}", "w"
        ) as f:
            json.dump(metadata, f)


class ModelsFactory:
    def __init__(self, model_type: Literal["flame", "smplx", "smal"]):
        self.model_type = model_type
        self.utils = Utils()

    def get_model(self, **kwargs) -> nn.Module:
        if self.model_type == "smplx":
            return self.utils.get_smplx_model(**kwargs)
        else:
            if "gender" in kwargs:
                kwargs.pop("gender")

            if self.model_type == "flame":
                return self.utils.get_flame_model(**kwargs)

            else:
                return self.utils.get_smal_model(**kwargs)

    def get_default_params(self, with_face: bool = False) -> Dict[str, torch.tensor]:

        params = {}

        if self.model_type == "smplx":
            params["body_pose"] = self.utils.get_default_parameters(body_pose=True)
            params["betas"] = self.utils.get_default_parameters()
            expression = None
            if with_face:
                expression = self.utils.get_default_face_expression()
            params["expression"] = expression

        elif self.model_type == "flame":
            params["shape_params"] = self.utils.get_default_face_shape()
            params["expression_params"] = self.utils.get_default_face_expression()

        else:
            params["beta"] = self.utils.get_default_parameters()

        return params

    def get_vt_ft(self):
        return self.utils.get_vt_ft(self.model_type)

    def get_renderer(self, py3d: bool = False, **kwargs) -> Union[Open3dRenderer, Pytorch3dRenderer]:
        if py3d:
            return Pytorch3dRenderer(**kwargs)
        return Open3dRenderer(**kwargs)

    def get_random_params(self, with_face: bool = False, rest_pose: bool = False) -> Dict[str, torch.tensor]:
        params = {}
        if self.model_type == "smplx":
            params["betas"] = self.utils.get_random_betas_smplx()
            if with_face:
                params["expression"] = self.utils.get_random_expression()
            else:
                params["expression"] = self.utils.get_default_parameters()
            if rest_pose:
                params["body_pose"] = self.utils.get_body_pose()
            else:
                params["body_pose"] = torch.eye(3).expand(1, 21, 3, 3)
        elif self.model_type == "flame":
            if with_face:
                params["expression_params"] = self.utils.get_random_expression_flame()
                params["shape_params"] = self.utils.get_default_face_shape()
            else:
                params["expression_params"] = self.utils.get_default_face_expression()
                params["shape_params"] = self.utils.get_random_shape()

        else:
            params["beta"] = self.utils.get_random_betas_smal()

        return params

    def get_key_name_for_model(self, with_face: bool = False) -> str:
        if self.model_type == "smplx":
            if with_face:
                return "expression"
            return "betas"
        elif self.model_type == "flame":
            if with_face:
                return "expression_params"
            return "shape_params"
        else:
            return "beta"


def plot_scatter_with_thumbnails(
    data_2d: np.ndarray,
    thumbnails: List[np.ndarray],
    labels: Optional[List[np.ndarray]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (1200, 1200),
    mark_size: int = 40,
):
    """
    Plot an interactive scatter plot with the provided thumbnails as tooltips.
    Args:
    - data_2d: 2D array of shape (n_samples, 2) containing the 2D coordinates of the data points.
    - thumbnails: List of thumbnails to be displayed as tooltips, each thumbnail should be a numpy array.
    - labels: List of labels to be used for coloring the data points, if None, no coloring is applied.
    - title: Title of the plot.
    - figsize: Size of the plot.
    - mark_size: Size of the data points.
    Returns:
    - Altair chart object.
    """

    def _return_thumbnail(img_array, size=100):
        """Return a thumbnail of the image array."""
        image = Image.fromarray(img_array)
        image.thumbnail((size, size), Image.ANTIALIAS)
        return image

    def _image_formatter(img):
        """Return a base64 encoded image."""
        with BytesIO() as buffer:
            img.save(buffer, "png")
            data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{data}"

    dataframe = pd.DataFrame(
        {
            "x": data_2d[:, 0],
            "y": data_2d[:, 1],
            "image": [_image_formatter(_return_thumbnail(thumbnail)) for thumbnail in thumbnails],
            "label": labels,
        }
    )

    chart = (
        alt.Chart(dataframe, title=title)
        .mark_circle(size=mark_size)
        .encode(
            x="x", y=alt.Y("y", axis=None), tooltip=["image"], color="label"
        )  # Must be a list for the image to render
        .properties(width=figsize[0], height=figsize[1])
        .configure_axis(grid=False)
        .configure_legend(orient="top", titleFontSize=20, labelFontSize=10, labelLimit=0)
    )

    if labels is not None:
        chart = chart.encode(color="label:N")

    return chart
