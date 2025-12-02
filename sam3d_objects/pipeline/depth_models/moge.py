# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
from loguru import logger
from .base import DepthModel


def get_local_moge_model_path():
    """
    Get the local MoGe model path if it exists.
    Checks for model in sam-3d-objects/models/moge/model.pt
    """
    # Try to find the models directory relative to this file
    current_file = os.path.abspath(__file__)
    # Navigate up from sam3d_objects/pipeline/depth_models/moge.py to sam-3d-objects/
    sam3d_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    local_model_path = os.path.join(sam3d_root, "models", "moge", "model.pt")
    
    if os.path.isfile(local_model_path):
        return local_model_path
    return None


def load_moge_model(pretrained_path: str = "Ruicheng/moge-vitl", device: str = "cuda"):
    """
    Load MoGe model from pretrained path or local cache.
    
    If local model exists at sam-3d-objects/models/moge/model.pt, weights will be loaded from there.
    Otherwise, the model will be downloaded from HuggingFace.
    
    This function can be used as a Hydra target:
        _target_: sam3d_objects.pipeline.depth_models.moge.load_moge_model
        pretrained_path: Ruicheng/moge-vitl
        device: cuda
    
    Args:
        pretrained_path: HuggingFace model path
        device: Device to load model on
    
    Returns:
        Loaded MoGe model
    """
    from moge.model.v1 import MoGeModel
    
    # Check for local model first
    local_model_path = get_local_moge_model_path()
    
    if local_model_path:
        logger.info(f"Loading MoGe model weights from local path: {local_model_path}")
        
        # Load checkpoint - may have nested structure {"model_config": ..., "model": state_dict}
        checkpoint = torch.load(local_model_path, map_location=device, weights_only=False)
        
        # Handle nested checkpoint format (model weights inside "model" key)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            logger.info("Detected nested checkpoint format, extracting model weights...")
            state_dict = checkpoint["model"]
            # If there's model_config, we can use it to verify compatibility
            if "model_config" in checkpoint:
                logger.info(f"Checkpoint includes model_config")
        else:
            # Direct state dict format
            state_dict = checkpoint
        
        # Create model with default config and load local weights
        # MoGe from_pretrained downloads both config and weights
        # We need to first get the model structure, then load local weights
        try:
            # Try loading with local_files_only first to avoid network call if already cached
            model = MoGeModel.from_pretrained(pretrained_path, local_files_only=True)
        except Exception:
            # If not cached, we need to download config at least once
            logger.info("Downloading MoGe model config (one-time)...")
            model = MoGeModel.from_pretrained(pretrained_path)
        
        # Load our local weights
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        logger.info("MoGe model loaded successfully from local weights")
        return model
    
    logger.info(f"Loading MoGe model from HuggingFace: {pretrained_path}")
    model = MoGeModel.from_pretrained(pretrained_path).to(device)
    model.eval()
    return model


def create_moge_depth_model(pretrained_path: str = "Ruicheng/moge-vitl", device: str = "cuda"):
    """
    Create a MoGe depth model wrapper for use in the inference pipeline.
    
    This function is designed to be used as a Hydra target for the depth_model config:
        depth_model:
            _target_: sam3d_objects.pipeline.depth_models.moge.create_moge_depth_model
            pretrained_path: Ruicheng/moge-vitl
            device: cuda
    
    Args:
        pretrained_path: HuggingFace model path
        device: Device to load model on
    
    Returns:
        MoGe depth model wrapper
    """
    model = load_moge_model(pretrained_path, device)
    return MoGe(model, device)


class MoGe(DepthModel):
    def __call__(self, image):
        output = self.model.infer(
            image.to(self.device), force_projection=False
        )
        pointmaps = output["points"]
        output["pointmaps"] = pointmaps
        return output