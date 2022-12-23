import cv2
import clip
import torch
import hydra
import random
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from typing import Union, Any, Dict, Tuple
from utils import Utils, Pytorch3dRenderer


class Image2Shape:
    def __init__(self, args, max_images_in_collage: int = 25):
        self.images_dir = Path(args.images_dir)
        self.utils = Utils()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = args.model_type
        self.with_face = args.with_face
        self.sliders_max_value = args.sliders_max_value
        if args.labels_weights is not None:
            self.labels_weights = torch.tensor(args.labels_weights).to(self.device)
        self.verbose = args.verbose
        self.display = args.display_images
        self.max_images_in_collage = max_images_in_collage
        self.num_img = 0  # num of collages saved
        if args.images_collage_path is not None:
            self.images_collage_path = Path(args.images_collage_path) / f"{self.num_img}.png"
            self.images_collage_list = []

        self._load_renderer(args.renderer_kwargs)
        self._load_model(args.model_path)
        self._load_clip_model()
        self._encode_labels()

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer = Pytorch3dRenderer(**kwargs)
        self.image_size = kwargs.img_size

    def _load_model(self, model_path: str):
        self.model, labels = self.utils.get_model_to_eval(model_path)
        self.labels = self._flatten_list_of_lists(labels)
        if not hasattr(self, "labels_weights"):
            self.labels_weights = torch.ones(len(self.labels)).to(self.device)

    def _load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_labels(self):
        self.encoded_labels = clip.tokenize(self.labels).to(self.device)

    @staticmethod
    def _flatten_list_of_lists(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    def _get_smplx_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        betas = pred_vec.cpu()
        smplx_out = self.utils.get_smplx_model(betas=betas)
        return smplx_out

    def _get_flame_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.with_face:
            flame_out = self.utils.get_flame_model(expression_params=pred_vec.cpu())
        else:
            flame_out = self.utils.get_flame_model(shape_params=pred_vec.cpu())
        return flame_out

    def _get_smal_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        smal_out = self.utils.get_smal_model(beta=pred_vec.cpu())
        return smal_out

    def get_render_mesh_kwargs(self, pred_vec: torch.Tensor) -> Dict[str, np.ndarray]:
        if self.model_type == "smplx":
            out = self._get_smplx_attributes(pred_vec=pred_vec)
        elif self.model_type == "flame":
            out = self._get_flame_attributes(pred_vec=pred_vec)
        else:
            out = self._get_smal_attributes(pred_vec=pred_vec)

        kwargs = {"verts": out[0], "faces": out[1], "vt": out[2], "ft": out[3]}

        return kwargs

    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        clip_min_value = 15
        clip_max_value = 25
        normalized_score = (scores - clip_min_value) / (clip_max_value - clip_min_value)
        normalized_score *= self.sliders_max_value
        normalized_score = torch.clamp(normalized_score, 0, self.sliders_max_value)
        normalized_score *= self.labels_weights
        return normalized_score.float()

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _save_images_collage(self, images: list):
        collage_shape = self.utils.get_plot_shape(len(images))[0]
        images_collage = []
        for i in range(collage_shape[0]):
            images_collage.append(np.hstack(images[i * collage_shape[1] : (i + 1) * collage_shape[1]]))
        images_collage = np.vstack([image for image in images_collage])
        cv2.imwrite(self.images_collage_path.as_posix(), images_collage)
        self.num_img += 1
        self.images_collage_path = self.images_collage_path.parent / f"{self.num_img}.png"

    def __call__(self):
        images_generator = list(self.images_dir.rglob("*.png"))
        random.shuffle(images_generator)
        for image_path in images_generator:

            image = Image.open(image_path.as_posix())
            encoded_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():

                clip_scores = self.clip_model(encoded_image, self.encoded_labels)[0]
                clip_scores = self.normalize_scores(clip_scores)
                shape_vector_pred = self.model(clip_scores)

            render_mesh_kwargs = self.get_render_mesh_kwargs(shape_vector_pred)

            rendered_img = self.renderer.render_mesh(**render_mesh_kwargs)
            rendered_img = self.adjust_rendered_img(rendered_img)

            input_image = cv2.imread(image_path.as_posix())
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, self.image_size)

            concatenated_img = np.concatenate((input_image, np.array(rendered_img)), axis=1)
            concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)

            # write clip scores on the image
            if self.verbose:
                for i, label in enumerate(self.labels):
                    cv2.putText(
                        concatenated_img,
                        f"{label}: {clip_scores[0][i].item():.2f}",
                        (370, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            if self.display:
                cv2.imshow("input", concatenated_img)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break

            if hasattr(self, "images_collage_path"):
                if len(self.images_collage_list) == self.max_images_in_collage:
                    self._save_images_collage(self.images_collage_list)
                    self.images_collage_list = []
                else:
                    self.images_collage_list.append(concatenated_img)


@hydra.main(config_path="config", config_name="image_to_shape")
def main(args: DictConfig):
    image2shape = Image2Shape(args)
    image2shape()


if __name__ == "__main__":
    main()
