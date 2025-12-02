import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask


def log_progress(message, fraction=None):
    if fraction is not None:
        print(f"[Progress {fraction*100:.1f}%] {message}")
    else:
        print(f"[Progress] {message}")

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42, progress_callback=log_progress)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
