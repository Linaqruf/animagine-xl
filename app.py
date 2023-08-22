#!/usr/bin/env python

from __future__ import annotations

import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import torch
import json
from huggingface_hub import hf_hub_download
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

DESCRIPTION = '# Animagine XL'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv(
    'CACHE_EXAMPLES') == '1'
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))
USE_TORCH_COMPILE = os.getenv('USE_TORCH_COMPILE') == '1'
ENABLE_CPU_OFFLOAD = os.getenv('ENABLE_CPU_OFFLOAD') == '1'

MODEL = "Linaqruf/animagine-xl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensors=True,
        variant='fp16')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet,
                                  mode='reduce-overhead',
                                  fullgraph=True)
else:
    pipe = None

with open("sdxl_loras.toml", "r") as file:
    data = toml.load(file)
    sdxl_loras = [
        {
            "image": get_image_path(item["image"]),
            "title": item["title"],
            "repo": item["repo"],
            "trigger_word": item.get("trigger_word", ""),
            "weights": item["weights"],
        }
        for item in data['data']
    ]
saved_names = [hf_hub_download(item["repo"], item["weights"]) for item in sdxl_loras]
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def get_image_path(base_path):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for ext in extensions:
        if os.path.exists(base_path + ext):
            return base_path + ext
    # If no match is found, return None or raise an error
    return None
    

def generate(prompt: str,
             negative_prompt: str = '',
             prompt_2: str = '',
             negative_prompt_2: str = '',
             use_prompt_2: bool = False,
             seed: int = 0,
             width: int = 1024,
             height: int = 1024,
             target_width: int = 1024,
             target_height: int = 1024,
             original_width: int = 4096,
             original_height: int = 4096,
             guidance_scale: float = 12.0,
             num_inference_steps: int = 50,
             use_lora: bool = False,
             lora_selection: str = "",
             lora_weight: float = 1.0,
             set_target_size: bool = False,
             set_original_size: bool = False) -> PIL.Image.Image:
    
    generator = torch.Generator().manual_seed(seed)

    cross_attention_kwargs = None

    if not set_original_size:
        original_width = 4096
        original_height = 4096
    if not set_target_size:
        target_width = width
        target_height = height   
    if negative_prompt == '':
        negative_prompt = None  # type: ignore
    if not use_prompt_2:
        prompt_2 = None  # type: ignore
        negative_prompt_2 = None  # type: ignore
    if negative_prompt_2 == '':
        negative_prompt_2 = None

    if use_lora and lora_selection:
        lora_index = [item["title"] for item in sdxl_loras].index(lora_selection)
        full_path_lora = saved_names[lora_index]  

        pipe.load_lora_weights(full_path_lora)
        cross_attention_kwargs = {"scale": lora_weight}
    else:
        pipe.unload_lora_weights()

    return pipe(prompt=prompt,
                    negative_prompt=negative_prompt,
                    prompt_2=prompt_2,
                    negative_prompt_2=negative_prompt_2,
                    width=width,
                    height=height,
                    target_size=(target_width, target_height),
                    original_size=(original_width, original_height),
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type='pil',
                    cross_attention_kwargs=cross_attention_kwargs,
                    ).images[0]

examples = [
    'face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck',
    'face focus, bishounen, masterpiece, best quality, 1boy, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck',
]

with open("lora.toml", "r") as file:
    data = toml.load(file)
    sdxl_loras = [
        {
            "image": get_image_path(item["image"]),
            "title": item["title"],
            "repo": item["repo"],
            "weights": item["weights"],
        }
        for item in data['data']
    ]
saved_names = [hf_hub_download(item["repo"], item["weights"]) for item in sdxl_loras]


with gr.Blocks(css='style.css', theme='NoCrypt/miku@1.2.1') as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value='Duplicate Space for private use',
        elem_id='duplicate-button',
        visible=os.getenv('SHOW_DUPLICATE_BUTTON') == '1'
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                prompt = gr.Text(
                    label='Prompt',
                    max_lines=5,
                    placeholder='Enter your prompt',
                )
                negative_prompt = gr.Text(
                    label='Negative Prompt',
                    max_lines=5,
                    placeholder='Enter a negative prompt',
                    value='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
                )

                with gr.Row():
                    use_prompt_2 = gr.Checkbox(
                        label='Use prompt 2', 
                        value=False
                    )   
                    use_lora = gr.Checkbox(
                        label='Use LoRA', 
                        value=False
                    )

            with gr.Group():
                prompt_2 = gr.Text(
                    label='Prompt 2',
                    max_lines=1,
                    placeholder='Enter your prompt',
                    visible=False,
                )
                negative_prompt_2 = gr.Text(
                    label='Negative prompt 2',
                    max_lines=1,
                    placeholder='Enter a negative prompt',
                    visible=False,
                )

            with gr.Group():
                lora_selection = gr.Dropdown(
                    choices=[item["title"] for item in sdxl_loras],
                    label="Select LoRA",
                    visible=False,
                )
                lora_weight = gr.Slider(
                    label="LoRA weight",
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=1,
                    visible=False
                )

            with gr.Group():
                with gr.Row():
                    width = gr.Slider(
                        label='Width',
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label='Height',
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    
                with gr.Accordion(label='Advanced Options', open=False):
                    seed = gr.Slider(
                        label='Seed',
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0
                    )
                    
                    randomize_seed = gr.Checkbox(
                        label='Randomize seed',
                        value=True
                    )
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label='Guidance scale',
                            minimum=1,
                            maximum=20,
                            step=0.1,
                            value=12.0)
                        num_inference_steps = gr.Slider(
                            label='Number of inference steps',
                            minimum=10,
                            maximum=100,
                            step=1,
                            value=50)
                    with gr.Group(): 
                        with gr.Row():
                            set_target_size = gr.Checkbox(
                                label='Target Size', 
                                value=False
                            )   
                            set_original_size = gr.Checkbox(
                                label='Original Size', 
                                value=False
                            )
                    with gr.Group(): 
                        with gr.Row():
                            original_width = gr.Slider(
                                label='Original Width',
                                minimum=1024,
                                maximum=4096,
                                step=32,
                                value=4096,
                                visible=False,
                            )
                            original_height = gr.Slider(
                                label='Original Height',
                                minimum=1024,
                                maximum=4096,
                                step=32,
                                value=4096,
                                visible=False,
                            )
                        with gr.Row():
                            target_width = gr.Slider(
                                label='Target Width',
                                minimum=1024,
                                maximum=4096,
                                step=32,
                                value=width.value,
                                visible=False,
                                )
                            target_height = gr.Slider(
                                label='Target Height',
                                minimum=1024,
                                maximum=4096,
                                step=32,
                                value=height.value,
                                visible=False,
                                )
        with gr.Column(scale=2):
            with gr.Blocks():
                run_button = gr.Button(
                    'Generate', 
                    variant='primary'
                )
            result = gr.Image(
                label='Result', 
                show_label=False
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES
    )
    
    use_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_prompt_2,
        outputs=prompt_2,
        queue=False,
        api_name=False,
    )
    use_prompt_2.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_prompt_2,
        outputs=negative_prompt_2,
        queue=False,
        api_name=False,
    )
    use_lora.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=use_lora,
        outputs=[lora_selection, lora_weight],
        queue=False,
        api_name=False,
    )
    set_target_size.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=set_target_size,
        outputs=[target_width, target_height],
        queue=False,
        api_name=False,
    )
    set_original_size.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=set_original_size,
        outputs=[original_width, original_height],
        queue=False,
        api_name=False,
    )
    width.change(
        fn=lambda x: gr.update(value=x),
        inputs=width,
        outputs=target_width,
        queue=False,
        api_name=False,
    )
    height.change(
        fn=lambda x: gr.update(value=x),
        inputs=height,
        outputs=target_height,
        queue=False,
        api_name=False,
    )   
    
    inputs = [
        prompt,
        negative_prompt,
        prompt_2,
        negative_prompt_2,
        use_prompt_2,
        seed,
        width,
        height,
        target_width,
        target_height,
        original_width,
        original_height,
        guidance_scale,
        num_inference_steps,
        use_lora,
        lora_selection,
        lora_weight,
        set_target_size,
        set_original_size
    ]
    prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name='run',
    )
    negative_prompt.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    prompt_2.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    negative_prompt_2.submit(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )
    run_button.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name=False,
    )

demo.queue(max_size=20).launch(debug=is_colab, share=is_colab)
