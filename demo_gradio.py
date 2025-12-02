# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Gradio Web Interface for SAM 3D Objects (Gradio 6.0+)
Run this script to launch a web interface for 3D object reconstruction.
"""

import os
import sys
import random
import argparse

# Configure PyTorch for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import threading
from datetime import datetime
import numpy as np
from PIL import Image
import gradio as gr
from gradio.themes.utils import fonts, colors, sizes
import torch

# NOTE: Removed multiprocessing.set_start_method('spawn') as Gradio uses threading, not multiprocessing
# Setting spawn mode was interfering with CUDA context in worker threads

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask, render_video, ready_gaussian_for_video_rendering

# Global variable for the model
inference_model = None

# Path to example images
EXAMPLES_BASE_PATH = os.path.join(os.path.dirname(__file__), "notebook", "images")
OUTPUTS_BASE_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_BASE_DIR, exist_ok=True)


def ensure_numpy_uint8(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.dtype != np.uint8:
        clipped = np.clip(image, 0, 255)
        image = clipped.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] >= 3:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def prepare_mask_array(mask_input, target_hw):
    if isinstance(mask_input, Image.Image):
        mask_arr = np.array(mask_input)
    else:
        mask_arr = np.array(mask_input)

    if mask_arr.ndim == 3:
        if mask_arr.shape[-1] == 4:
            mask = mask_arr[..., -1] > 0
        else:
            mask = mask_arr[..., 0] > 0
    else:
        mask = mask_arr > 0

    if mask.shape != target_hw:
        pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
        pil_mask = pil_mask.resize((target_hw[1], target_hw[0]), Image.NEAREST)
        mask = np.array(pil_mask) > 0
    return mask


def get_next_output_dir():
    os.makedirs(OUTPUTS_BASE_DIR, exist_ok=True)
    existing = [
        int(name)
        for name in os.listdir(OUTPUTS_BASE_DIR)
        if name.isdigit()
    ]
    next_index = max(existing) + 1 if existing else 1
    while True:
        dir_name = f"{next_index:04d}"
        dir_path = os.path.join(OUTPUTS_BASE_DIR, dir_name)
        try:
            os.makedirs(dir_path)
            return dir_path, dir_name
        except FileExistsError:
            next_index += 1


def save_inputs(output_dir, image, mask):
    image_path = os.path.join(output_dir, "input.png")
    Image.fromarray(image).save(image_path)
    mask_path = os.path.join(output_dir, "mask.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    return image_path, mask_path


def write_metadata(output_dir, metadata):
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    return metadata_path


def load_model():
    """Load the inference model (only once)"""
    global inference_model
    if inference_model is None:
        print("Loading SAM 3D Objects model...")
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        inference_model = Inference(config_path, compile=False)
        print("Model loaded successfully!")
    return inference_model


def process_image(
    image,
    mask_image,
    seed,
    random_seed_enabled,
    stage1_steps,
    stage1_cfg,
    stage2_steps,
    stage2_cfg,
    mesh_postprocess,
    texture_baking,
    layout_postprocess,
    vertex_colors,
    simplify_ratio,
    texture_size,
    video_resolution,
    video_frames,
    video_radius,
    video_fov,
    video_pitch,
    video_yaw,
    progress=gr.Progress(),
):
    """
    Process an image with a mask to generate 3D reconstruction.
    
    Args:
        image: Input RGB image (numpy array or PIL Image)
        mask_image: Binary mask image (numpy array or PIL Image)  
        seed: Random seed for reproducibility
        random_seed_enabled: Whether to generate a random seed
        stage1_steps: Diffusion steps for sparse structure sampling
        stage1_cfg: CFG strength for Stage 1
        stage2_steps: Diffusion steps for latent sampling
        stage2_cfg: CFG strength for Stage 2
        mesh_postprocess: Whether to run mesh clean-up
        texture_baking: Whether to bake textures
        layout_postprocess: Whether to run layout optimization
        vertex_colors: Whether to preserve vertex colors
        simplify_ratio: Ratio of triangles to remove in mesh simplification
        texture_size: Resolution of baked texture
        video_resolution: Resolution of preview video frames
        video_frames: Number of frames for preview video
        video_radius: Camera radius for orbit
        video_fov: Camera field-of-view
        video_pitch: Pitch angle (degrees)
        video_yaw: Starting yaw angle (degrees)
        progress: Gradio Progress tracker (injected automatically)
    
    Returns:
        Tuple of (ply_file_path, model3d_path, video_path, metadata_path, status_message, last_seed)
    """
    def progress_callback(message, fraction=None):
        """Simplified progress callback - just prints to console"""
        try:
            if fraction is not None:
                pct = max(0.0, min(1.0, float(fraction)))
                print(f"[Progress {pct*100:.1f}%] {message}", flush=True)
                # Try to update Gradio UI, but don't let it block
                try:
                    progress(pct, desc=message)
                except:
                    pass
            else:
                print(f"[Progress] {message}", flush=True)
        except Exception as e:
            # Don't let progress update errors crash the inference
            print(f"Warning: Progress callback error: {e}", flush=True)

    try:
        # Load model if not already loaded
        model = load_model()
        
        # Simple CUDA device info (no excessive synchronization)
        if torch.cuda.is_available():
            print(f"[DEBUG] CUDA device: {torch.cuda.current_device()}", flush=True)
            print(f"[DEBUG] CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"[DEBUG] Thread: {threading.current_thread().name}", flush=True)
            print(f"[DEBUG] Thread ID: {threading.get_ident()}", flush=True)
        
        progress_callback("Preparing inputs", 0.01)
        
        # Convert inputs to numpy arrays that match the CLI utilities
        if image is None:
            return None, None, None, None, "❌ Error: Please provide an input image", ""
        image = ensure_numpy_uint8(image)
        
        # Handle mask
        if mask_image is None:
            return None, None, None, None, "❌ Error: Please provide a mask image", ""
        mask = prepare_mask_array(mask_image, image.shape[:2])
        mask_coverage = float(mask.mean())
        mask_warnings = []
        if mask_coverage < 0.001:
            mask_warnings.append(
                "Mask looks empty; the reconstruction may fail. Double-check the mask."
            )
        elif mask_coverage > 0.95:
            mask_warnings.append(
                "Mask covers most of the image; results may be washed out. Consider a tighter mask."
            )
        
        print(
            f"Processing image dtype={image.dtype} shape={image.shape} "
            f"and mask dtype={mask.dtype} shape={mask.shape}",
            flush=True
        )
        print(f"Using seed: {seed}", flush=True)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        output_dir, run_id = get_next_output_dir()
        # Save inputs for metadata/record keeping
        input_image_path, mask_path = save_inputs(output_dir, image, mask)
        timings["save_inputs"] = time.perf_counter() - t0

        progress_callback("Running inference", 0.05)
        # Generate random seed if checkbox is enabled
        if random_seed_enabled:
            seed_value = random.randint(0, 2**31 - 1)
            print(f"Generated random seed: {seed_value}", flush=True)
        else:
            seed_value = int(seed)
        stage1_steps_val = int(stage1_steps) if stage1_steps else None
        stage1_cfg_val = float(stage1_cfg) if stage1_cfg else None
        stage2_steps_val = int(stage2_steps) if stage2_steps else None
        stage2_cfg_val = float(stage2_cfg) if stage2_cfg else None
        simplify_ratio_val = float(simplify_ratio)
        texture_size_val = int(texture_size)
        video_resolution = int(video_resolution)
        video_frames = int(video_frames)
        video_radius = float(video_radius)
        video_fov = float(video_fov)
        video_pitch = float(video_pitch)
        video_yaw = float(video_yaw)
        user_params = {
            "stage1_steps": stage1_steps_val,
            "stage1_cfg": stage1_cfg_val,
            "stage2_steps": stage2_steps_val,
            "stage2_cfg": stage2_cfg_val,
            "mesh_postprocess": bool(mesh_postprocess),
            "texture_baking": bool(texture_baking),
            "layout_postprocess": bool(layout_postprocess),
            "vertex_colors": bool(vertex_colors),
            "simplify_ratio": simplify_ratio_val,
            "texture_size": texture_size_val,
        }
        
        # Log pre-inference state (minimal CUDA calls)
        if torch.cuda.is_available():
            print(f"[DEBUG] Pre-inference GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
        
        # Run inference directly with numpy arrays (like demo.py does)
        print(f"[DEBUG] Starting inference with image shape={image.shape}, mask shape={mask.shape}", flush=True)
        print(f"[DEBUG] Image dtype: {image.dtype}, mask dtype: {mask.dtype}", flush=True)
        print(f"[DEBUG] Mask unique values: {np.unique(mask)}", flush=True)
        
        print(f"[DEBUG] About to call model() - Thread: {threading.current_thread().name}", flush=True)
        print(f"[DEBUG] Model type: {type(model)}", flush=True)
        print(f"[DEBUG] Calling model with seed={seed_value}", flush=True)
        
        t0 = time.perf_counter()
        output = model(
            image,  # Use direct numpy array
            mask,   # Use direct boolean mask
            seed=seed_value,
            stage1_inference_steps=stage1_steps_val,
            stage2_inference_steps=stage2_steps_val,
            stage1_cfg_strength=stage1_cfg_val,
            stage2_cfg_strength=stage2_cfg_val,
            with_mesh_postprocess=mesh_postprocess,
            with_texture_baking=texture_baking,
            with_layout_postprocess=layout_postprocess,
            use_vertex_color=vertex_colors,
            simplify_ratio=simplify_ratio_val,
            texture_size=texture_size_val,
            progress_callback=progress_callback,
        )
        
        print(f"[DEBUG] model() call returned successfully!", flush=True)
        timings["inference"] = time.perf_counter() - t0
        print(f"[DEBUG] Inference completed in {timings['inference']:.2f}s", flush=True)
        
        # Log post-inference state
        if torch.cuda.is_available():
            print(f"[DEBUG] Post-inference GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)

        # Save PLY file
        progress_callback("Saving reconstruction", 0.9)
        t0 = time.perf_counter()
        ply_path = os.path.join(output_dir, "reconstruction.ply")
        output["gs"].save_ply(ply_path)
        timings["save_outputs"] = time.perf_counter() - t0
        
        # Generate preview video / 3D preview path
        model3d_path = ply_path
        video_path = None
        try:
            print("Generating preview video...")
            from inference import make_scene
            
            # Create scene from output
            scene_gs = make_scene(output)
            scene_gs = ready_gaussian_for_video_rendering(scene_gs)
            
            progress_callback("Rendering preview video", 0.95)
            video_result = render_video(
                scene_gs,
                resolution=video_resolution,
                bg_color=(1, 1, 1),  # White background
                num_frames=video_frames,
                r=video_radius,
                fov=video_fov,
                pitch_deg=video_pitch,
                yaw_start_deg=video_yaw,
            )
            
            frames = video_result["color"]
            # render_frames returns uint8 frames already in [0, 255] range
            frames_uint8 = frames
            video_path = os.path.join(output_dir, "preview.mp4")
            import imageio

            try:
                with imageio.get_writer(
                    video_path,
                    format="FFMPEG",
                    mode="I",
                    fps=30,
                    codec="libx264",
                ) as writer:
                    for frame in frames_uint8:
                        writer.append_data(frame)
                print(f"Preview video saved to {video_path}")
            except Exception as ffmpeg_err:
                print(f"Warning: FFMPEG writer not available ({ffmpeg_err}); skipping video.")
                video_path = None
            
        except Exception as e:
            print(f"Warning: Could not generate preview video: {e}")
            video_path = None

        metadata = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": seed_value,
            "image_shape": list(image.shape),
            "mask_shape": list(mask.shape),
            "mask_coverage": mask_coverage,
            "parameters": {
                **user_params,
                "video": {
                    "resolution": video_resolution,
                    "frames": video_frames,
                    "radius": video_radius,
                    "fov": video_fov,
                    "pitch_deg": video_pitch,
                    "yaw_start_deg": video_yaw,
                },
            },
            "paths": {
                "output_dir": output_dir,
                "ply": ply_path,
                "video": video_path,
                "input_image": input_image_path,
                "mask": mask_path,
            },
            "timings": timings,
        }
        metadata_path = write_metadata(output_dir, metadata)

        progress_callback("Done", 1.0)
        status_message = f"✅ Reconstruction complete! Files saved to outputs/{run_id}"
        if mask_warnings:
            status_message += "\n⚠️ " + " ".join(mask_warnings)
        return ply_path, model3d_path, video_path, metadata_path, status_message, str(seed_value)
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, None, error_msg, ""


def get_example_data():
    """Get example image/mask pairs for the Gradio Examples component"""
    examples = []
    
    # Example 1: Kids room - shelf/furniture (mask 14)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "shutterstock_stylish_kidsroom_1640806567", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "shutterstock_stylish_kidsroom_1640806567", "14.png"),
        42
    ])
    
    # Example 2: Kids room - different object (mask 5)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "shutterstock_stylish_kidsroom_1640806567", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "shutterstock_stylish_kidsroom_1640806567", "5.png"),
        42
    ])
    
    # Example 3: Human object
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "human_object", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "human_object", "0.png"),
        42
    ])
    
    # Example 4: Living room (mask 0)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "137444513_Livingroom-graphic81", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "137444513_Livingroom-graphic81", "0.png"),
        42
    ])
    
    # Example 5: Living room (mask 5)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "137444513_Livingroom-graphic81", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "137444513_Livingroom-graphic81", "5.png"),
        42
    ])
    
    # Example 6: Kid box (mask 0)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "kid_box", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "kid_box", "0.png"),
        42
    ])
    
    # Example 7: Modern colorful interior (mask 10)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "shutterstock_modern_colorful_Interior_2620125197", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "shutterstock_modern_colorful_Interior_2620125197", "10.png"),
        42
    ])
    
    # Example 8: Wild animals waterhole (mask 0)
    examples.append([
        os.path.join(EXAMPLES_BASE_PATH, "id3_shutterstock_WildAnimal_Waterhole_2010559391", "image.png"),
        os.path.join(EXAMPLES_BASE_PATH, "id3_shutterstock_WildAnimal_Waterhole_2010559391", "0.png"),
        42
    ])
    
    # Filter to only existing examples
    valid_examples = []
    for ex in examples:
        if os.path.exists(ex[0]) and os.path.exists(ex[1]):
            valid_examples.append(ex)
    
    return valid_examples


# Create the Gradio interface
def create_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(
        title="SAM 3D Objects - 3D Reconstruction",
    ) as demo:
        gr.Markdown("### SAM 3D Objects - V1 : https://www.patreon.com/posts/144931360")
        gr.Markdown("#### Transform 2D images into 3D Gaussian Splat reconstructions")
       
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input Image")
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                
                gr.Markdown("### 🎭 Mask Image")
                gr.Markdown("*Upload a binary mask where white (255) indicates the object to reconstruct*")
                mask_image = gr.Image(
                    label="Upload Mask",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    image_mode="RGBA",
                )
                
                with gr.Row():
                    seed_input = gr.Number(
                        label="🎲 Random Seed",
                        value=42,
                        precision=0,
                        scale=3,
                    )
                    random_seed_checkbox = gr.Checkbox(
                        label="Randomize",
                        value=False,
                        scale=1,
                    )
                
                last_seed_display = gr.Textbox(
                    label="Last Used Seed",
                    value="",
                    interactive=False,
                    visible=True,
                )

                with gr.Accordion("⚙️ Advanced Model Settings", open=False):
                    gr.Markdown("**Diffusion Sampling**")
                    stage1_steps = gr.Slider(
                        label="Stage 1 Steps (Sparse Structure)",
                        minimum=5,
                        maximum=50,
                        value=25,
                        step=1,
                        info="Number of diffusion steps for sparse structure sampling",
                    )
                    stage1_cfg = gr.Slider(
                        label="Stage 1 CFG Strength",
                        minimum=0.0,
                        maximum=15.0,
                        value=7.0,
                        step=0.5,
                        info="Classifier-free guidance strength for Stage 1 (default: 7.0)",
                    )
                    stage2_steps = gr.Slider(
                        label="Stage 2 Steps (Latent Sampling)",
                        minimum=5,
                        maximum=50,
                        value=25,
                        step=1,
                        info="Number of diffusion steps for latent sampling",
                    )
                    stage2_cfg = gr.Slider(
                        label="Stage 2 CFG Strength",
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                        info="Classifier-free guidance strength for Stage 2 (default: 1.0 from config)",
                    )
                    
                    gr.Markdown("**Mesh Processing**")
                    mesh_checkbox = gr.Checkbox(
                        label="Mesh Postprocess",
                        value=False,
                        info="Apply mesh cleanup (simplification, smoothing)",
                    )
                    texture_checkbox = gr.Checkbox(
                        label="Texture Baking",
                        value=False,
                        info="Bake textures from Gaussian splats to mesh",
                    )
                    layout_checkbox = gr.Checkbox(
                        label="Layout Postprocess",
                        value=True,
                        info="Run layout optimization for pose refinement",
                    )
                    vertex_checkbox = gr.Checkbox(
                        label="Use Vertex Colors",
                        value=True,
                        info="Preserve vertex colors instead of using textures",
                    )
                    simplify_ratio = gr.Slider(
                        label="Mesh Simplify Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        info="Ratio of triangles to remove in simplification (0.95 = remove 95%)",
                    )
                    texture_size_slider = gr.Slider(
                        label="Texture Size",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=256,
                        info="Resolution of baked texture (only used if Texture Baking is enabled)",
                    )

                with gr.Accordion("🎞️ Video Preview Settings", open=False):
                    video_resolution = gr.Slider(
                        label="Resolution (px)",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                    )
                    video_frames = gr.Slider(
                        label="Frames",
                        minimum=30,
                        maximum=180,
                        value=60,
                        step=10,
                    )
                    video_radius = gr.Slider(
                        label="Camera Radius",
                        minimum=0.5,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                    )
                    video_fov = gr.Slider(
                        label="Field of View",
                        minimum=20,
                        maximum=90,
                        value=60,
                        step=1,
                    )
                    video_pitch = gr.Slider(
                        label="Pitch (deg)",
                        minimum=-45,
                        maximum=45,
                        value=15,
                        step=1,
                    )
                    video_yaw = gr.Slider(
                        label="Start Yaw (deg)",
                        minimum=-180,
                        maximum=180,
                        value=-45,
                        step=1,
                    )
                
                process_btn = gr.Button(
                    "🚀 Generate 3D Reconstruction",
                    variant="primary",
                    size="lg",
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Output")
                ply_output = gr.File(
                    label="Download PLY File",
                )
                model_output = gr.Model3D(
                    label="3D Preview",
                )
                video_output = gr.Video(
                    label="Preview Video (Turntable)",
                )
                metadata_output = gr.File(
                    label="Metadata JSON",
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )
        
        gr.Markdown("---")
        
        # Examples section
        gr.Markdown("### 🖼️ Click an Example to Load")
        gr.Examples(
            examples=get_example_data(),
            inputs=[input_image, mask_image, seed_input],
            label="Example Images & Masks",
            examples_per_page=8,
        )
        
        gr.Markdown("---")
        
        with gr.Accordion("📖 Instructions", open=False):
            gr.Markdown("""
            ### How to use:
            1. **Click an example** below OR **upload your own image and mask**
            2. **Upload a mask** - A binary image where:
               - **White (255)** = the object you want to reconstruct
               - **Black (0)** = background to ignore
            3. **Set a seed** (optional) - For reproducible results
            4. **Click Generate** - Wait for the reconstruction to complete
            5. **Download the result** - Get the PLY file for use in 3D software
            
            ### Tips:
            - The mask should tightly cover the object of interest
            - Clean masks with clear boundaries work best
            - Processing may take 30-60 seconds depending on your GPU
            """)
        
        # Connect the button to the processing function
        # NOTE: queue=False is CRITICAL for CUDA GPU inference to work properly
        # See GRADIO_BUG_REPORT.md for details on this Gradio threading issue
        process_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                mask_image,
                seed_input,
                random_seed_checkbox,
                stage1_steps,
                stage1_cfg,
                stage2_steps,
                stage2_cfg,
                mesh_checkbox,
                texture_checkbox,
                layout_checkbox,
                vertex_checkbox,
                simplify_ratio,
                texture_size_slider,
                video_resolution,
                video_frames,
                video_radius,
                video_fov,
                video_pitch,
                video_yaw,
            ],
            outputs=[
                ply_output,
                model_output,
                video_output,
                metadata_output,
                status_output,
                last_seed_display,
            ],
            queue=False,  # Bypass queue threading - fixes CUDA hang issue
        )
    
    return demo


# Create a custom theme with highly readable fonts
def create_readable_theme():
    """Create a theme with Lexend font - designed for readability"""
    return gr.themes.Soft(
        primary_hue=colors.indigo,
        secondary_hue=colors.purple,
        neutral_hue=colors.slate,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_md,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Lexend"),  # Lexend is specifically designed for readability
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    )


# Custom CSS for better readability
CUSTOM_CSS = """
/* Increase base font size for better readability */
.gradio-container {
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Make labels more prominent */
label {
    font-weight: 500 !important;
    font-size: 1rem !important;
}

/* Better button styling */
.primary {
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 1.5rem !important;
}

/* Markdown content readability */
.prose {
    font-size: 1rem !important;
    line-height: 1.7 !important;
}

.prose h3 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
}

/* Status text readability */
textarea {
    font-size: 1rem !important;
    line-height: 1.5 !important;
}

/* Accordion headers */
.accordion-header {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* Examples gallery styling */
.gallery {
    gap: 8px !important;
}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3D Objects - Gradio Web Interface")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM 3D Objects - Gradio Web Interface")
    print("=" * 60)
    
    # Pre-load the model
    print("\nInitializing model (this may take a minute)...")
    load_model()
    
    # Create and launch the interface
    demo = create_interface()
    
    # NOTE: The main process_image function uses queue=False to bypass queue threading
    # This is required because Gradio's queue worker threads cause CUDA operations to hang
    # See GRADIO_BUG_REPORT.md for full details on this issue
    # The queue is still configured here for any other potential event handlers
    demo.queue(
        default_concurrency_limit=1,
        max_size=None,
    )
    
    print("\n" + "=" * 60)
    print("Starting Gradio server...")
    if args.share:
        print("Public sharing enabled - a shareable link will be created")
    print("=" * 60 + "\n")
    
    # In Gradio 6.0+, theme and css are passed to launch()
    demo.launch(
        inbrowser=True,  # Auto-open browser
        share=args.share,  # Create a public link if --share is provided
        show_error=True,
        theme=create_readable_theme(),
        css=CUSTOM_CSS,
    )
