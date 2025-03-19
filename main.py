import os
import time
import threading
import torch
import gradio as gr
from firebase_admin import credentials, initialize_app

cred = credentials.Certificate("firebase-key.json")
initialize_app(cred)

model_cache = {}

def load_model(name):
    if name not in model_cache:
        model_cache[name] = None
        threading.Thread(target=_actual_loading,args=(name,)).start()
    while model_cache[name] is None:
        time.sleep(0.1)
    return model_cache[name]

def _actual_loading(name):
    if name == "sdxl":
        from diffusers import StableDiffusionXLImg2ImgPipeline
        model_cache[name] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).cpu()
    elif name == "upscaler":
        from diffusers import StableDiffusionUpscalePipeline
        model_cache[name] = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16
        ).cpu()

def create_ui():
    with gr.Blocks(title="HyperReal AI", css="style.css") as ui:
        gr.Markdown("# 🎨 HyperReal Image Generator")
        with gr.Row():
            prompt = gr.Textbox(label="Describe your image")
            generate_btn = gr.Button("Generate", variant="primary")
        output_image = gr.Image(label="Result")
        generate_btn.click(
            fn=lambda p: load_model("sdxl")(prompt=p).images[0],
            inputs=prompt,
            outputs=output_image
        )
    return ui

if __name__ == "__main__":
    torch.set_num_threads(1)
    app = create_ui()
    app.launch(server_port=7860)