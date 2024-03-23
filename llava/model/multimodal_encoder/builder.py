import os
from .clip_encoder import CLIPVisionTower

try:
    from .dino_encoder import Dinov2VisionTower
    from .fusion_encoder import FusionVisionTower
except:
    pass


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if isinstance(vision_tower, str):
        vision_tower = [vision_tower]
    for path in vision_tower:
        assert os.path.exists(path)
    if len(vision_tower) == 1:
        vision_tower = vision_tower[0]
        if "clip" in vision_tower or "CLIP" in vision_tower:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if "dino" in vision_tower:
            return Dinov2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        raise NotImplementedError
    else:
        print("use fusion vision tower {}".format(vision_tower))
        return FusionVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
