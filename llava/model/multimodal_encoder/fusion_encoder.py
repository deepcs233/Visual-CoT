import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
    Dinov2Model,
    Dinov2Config,
    AutoImageProcessor,
    AutoModel,
)


class FusionVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        assert isinstance(vision_tower, list) and len(vision_tower) == 2
        self.vision_tower_name_0, self.vision_tower_name_1 = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.ft_vision_tower = getattr(args, "ft_vision_tower", False)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only_0 = CLIPVisionConfig.from_pretrained(self.vision_tower_name_0)
            self.cfg_only_1 = Dinov2Config.from_pretrained(self.vision_tower_name_1)

    def load_model(self):
        self.image_processor_0 = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name_0
        )
        self.image_processor_1 = AutoImageProcessor.from_pretrained(
            self.vision_tower_name_1
        )
        self.vision_tower_0 = CLIPVisionModel.from_pretrained(self.vision_tower_name_0)
        self.vision_tower_1 = Dinov2Model.from_pretrained(self.vision_tower_name_1)

        if not self.ft_vision_tower:
            self.vision_tower_0.requires_grad_(False)
            self.vision_tower_1.requires_grad_(False)
        else:
            self.vision_tower_0.requires_grad_(True)
            self.vision_tower_1.requires_grad_(True)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if self.ft_vision_tower:
            return self.forward_func(images)
        else:
            with torch.no_grad():
                return self.forward_func(images)

    def forward_func(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out_0 = self.vision_tower_0(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_forward_out_1 = self.vision_tower_1(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )

                feature0 = self.feature_select(image_forward_out_0).to(image.dtype)
                feature1 = self.feature_select(image_forward_out_1).to(image.dtype)

                image_feature = torch.cat((feature0, feature1), dim=-1)

                image_features.append(image_feature)
        else:
            _, _, h, w = images.shape
            images = images.reshape(-1, 2, 3, h, w).transpose(0, 1)
            images_0, images_1 = images[0], images[1]
            image_forward_outs_0 = self.vision_tower_0(
                images_0.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_forward_outs_1 = self.vision_tower_1(
                images_1.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            feature0 = self.feature_select(image_forward_outs_0).to(images.dtype)
            feature1 = self.feature_select(image_forward_outs_1).to(images.dtype)
            image_features = torch.cat((feature0, feature1), dim=-1)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower_0.dtype

    @property
    def device(self):
        return self.vision_tower_0.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower_0.config
        else:
            return self.cfg_only_0

    @property
    def hidden_size(self):
        if self.is_loaded:
            return (
                self.vision_tower_0.config.hidden_size
                + self.vision_tower_1.config.hidden_size
            )
        else:
            return self.cfg_only_0.hidden_size + self.cfg_only_1.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
