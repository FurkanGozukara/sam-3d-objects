# Use the EXACT same import path as demo.py
import sys
sys.path.append("notebook")
from inference import Inference, load_image

import os
import tempfile
import numpy as np
from PIL import Image
import gradio as gr

# Global inference instance
inference = None


def load_model():
    global inference
    if inference is None:
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        print("Loading model...")
        inference = Inference(config_path, compile=False)
        print("Model loaded!")
    return inference


def process_images(input_image, mask_image, seed):
    global inference
    
    if input_image is None:
        raise gr.Error("Please upload an input image")
    if mask_image is None:
        raise gr.Error("Please upload a mask image")
    
    if inference is None:
        raise gr.Error("Model not loaded")
    
    # Convert to numpy
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    if isinstance(mask_image, Image.Image):
        mask_image = np.array(mask_image)
    
    # Process mask like load_mask does
    mask = mask_image > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    
    image = input_image.astype(np.uint8)
    seed_value = int(seed) if seed else 42
    
    progress_messages = []
    def log_progress(message, fraction=None):
        if fraction is not None:
            progress_messages.append(f"[{fraction*100:.1f}%] {message}")
        else:
            progress_messages.append(message)
        print(f"Progress: {message}" + (f" ({fraction*100:.1f}%)" if fraction else ""))
    
    print(f"Running inference with seed={seed_value}")
    output = inference(image, mask, seed=seed_value, progress_callback=log_progress)
    
    output_path = tempfile.mktemp(suffix=".ply")
    output["gs"].save_ply(output_path)
    print(f"Saved to {output_path}")
    
    return output_path, "\n".join(progress_messages)


def create_app():
    with gr.Blocks(title="SAM 3D Objects") as demo:
        gr.Markdown("# SAM 3D Objects")
        
        with gr.Row():
            input_image = gr.Image(label="Input Image", type="numpy", height=350)
            mask_image = gr.Image(label="Mask Image", type="numpy", height=350)
        
        with gr.Row():
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            process_btn = gr.Button("Process", variant="primary")
        
        with gr.Row():
            output_file = gr.File(label="Download PLY")
            progress_log = gr.Textbox(label="Progress", lines=8)
        
        model_viewer = gr.Model3D(label="3D Preview")
        
        process_btn.click(
            fn=process_images,
            inputs=[input_image, mask_image, seed_input],
            outputs=[output_file, progress_log],
            queue=False,  # Run synchronously, not in queue thread
        ).then(
            fn=lambda x: x,
            inputs=[output_file],
            outputs=[model_viewer],
            queue=False,
        )
    
    return demo


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Model ready!")
    
    app = create_app()
    
    # Disable queue to run in main thread (avoids threading issues with CUDA)
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
