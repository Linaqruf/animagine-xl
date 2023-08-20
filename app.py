#!/usr/bin/env python

from __future__ import annotations

import os
import random

import gradio as gr
import numpy as np
import PIL.Image
import torch
import utils
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler

is_colab = utils.is_google_colab()

DESCRIPTION = '# Animagine XL'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv(
    'CACHE_EXAMPLES') == '1'
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '2048'))
USE_TORCH_COMPILE = os.getenv('USE_TORCH_COMPILE') == '1'
ENABLE_CPU_OFFLOAD = os.getenv('ENABLE_CPU_OFFLOAD') == '1'
SD_XL_BASE_RATIOS  = {
    "Vertical (9:16)": (768, 1344),
    "Portrait (4:5)": (912, 1144),
    "Square (1:1)": (1024, 1024),
    "Photo (4:3)": (1184, 888),
    "Landscape (3:2)": (1256, 832),
    "Widescreen (16:9)": (1368, 768),
    "Cinematic (21:9)": (1568, 672),
}

ASPECT_RATIOS = {str(v[0])+'×'+str(v[1]):v for k, v in SD_XL_BASE_RATIOS.items()}

MODEL = "Linaqruf/animagine-xl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    TXT2IMG_PIPE = StableDiffusionXLPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16')
    IMG2IMG_PIPE = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16')

    TXT2IMG_PIPE.scheduler = EulerAncestralDiscreteScheduler.from_config(TXT2IMG_PIPE.scheduler.config)
    IMG2IMG_PIPE.scheduler = EulerAncestralDiscreteScheduler.from_config(IMG2IMG_PIPE.scheduler.config)
       
    if ENABLE_CPU_OFFLOAD:
        TXT2IMG_PIPE.enable_model_cpu_offload()
        IMG2IMG_PIPE.enable_model_cpu_offload()
    else:
        TXT2IMG_PIPE.to(device)
        IMG2IMG_PIPE.to(device)

    if USE_TORCH_COMPILE:
        TXT2IMG_PIPE.unet = torch.compile(TXT2IMG_PIPE.unet,
                                  mode='reduce-overhead',
                                  fullgraph=True)
        IMG2IMG_PIPE.unet = torch.compile(IMG2IMG_PIPE.unet,
                                  mode='reduce-overhead',
                                  fullgraph=True)
        
else:
    TXT2IMG_PIPE = None
    IMG2IMG_PIPE = None


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def resize_images_by_percentage(imgs, percentage, min_dimension=1024):
    resized = []
    for img in imgs:
        # Calculate the new dimensions
        new_width = int(img.width * percentage / 100)
        new_height = int(img.height * percentage / 100)
        
        if new_width < min_dimension and new_width > new_height:
            ratio = min_dimension / new_width
            new_width = min_dimension
            new_height = int(new_height * ratio)
        elif new_height < min_dimension and new_height >= new_width:
            ratio = min_dimension / new_height
            new_height = min_dimension
            new_width = int(new_width * ratio)

        r_img = img.resize((new_width, new_height), PIL.Image.LANCZOS)
        if hasattr(img, "filename"): 
            r_img.filename = img.filename
        resized.append(r_img)
    return resized



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
             guidance_scale_base: float = 12.0,
             num_inference_steps_base: int = 50,
             image = None,
             strength: float = 0.55,
             init_image: bool = False,
             aspect_ratio_presets: bool = False,
             aspect_ratio_selection = None) -> PIL.Image.Image:
    
    generator = torch.Generator().manual_seed(seed)

    if negative_prompt == '':
        negative_prompt = None  # type: ignore
    if not use_prompt_2:
        prompt_2 = None  # type: ignore
        negative_prompt_2 = None  # type: ignore
    if negative_prompt_2 == '':
        negative_prompt_2 = None  
    if aspect_ratio_presets == True: 
        if aspect_ratio_selection not in ASPECT_RATIOS:
            aspect_ratio_selection = next(iter(ASPECT_RATIOS))  # default to the first option
        width, height = ASPECT_RATIOS[aspect_ratio_selection]

    if image is None:
        return TXT2IMG_PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            width=width,
            height=height,
            target_size=(target_width, target_height),
            original_size=(original_width, original_height),
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type='pil',
        ).images[0]
    else:
        if init_image:
            downscale_ratio = 100
            image = resize_images_by_percentage([image], downscale_ratio)[0]
            
        return IMG2IMG_PIPE(
            prompt=prompt,
            image=image,
            strength=strength,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            target_size=(target_width, target_height),
            original_size=(original_width, original_height),
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type='pil',
        ).images[0]        


examples = [
    'face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck',
    'face focus, bishounen, masterpiece, best quality, 1boy, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck',
]

with gr.Blocks(css='style.css', theme='NoCrypt/miku@1.2.1') as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value='Duplicate Space for private use',
                       elem_id='duplicate-button',
                       visible=os.getenv('SHOW_DUPLICATE_BUTTON') == '1')
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Text(
                label='Prompt',
                max_lines=2,
                placeholder='Enter your prompt',
            )
            negative_prompt = gr.Text(
                label='Negative Prompt',
                max_lines=2,
                placeholder='Enter a negative prompt',
                value='lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
            )
            with gr.Row():
                use_prompt_2 = gr.Checkbox(
                    label='Use prompt 2', 
                    value=False
                )
                init_image = gr.Checkbox(
                    label='Init Image', 
                    value=False
                )
                aspect_ratio_presets = gr.Checkbox(
                    label='Aspect Ratio', 
                    value=False
                )

            prompt_2 = gr.Text(
                label='Prompt 2',
                max_lines=3,
                placeholder='Enter your prompt',
                visible=False,
            )
            negative_prompt_2 = gr.Text(
                label='Negative prompt 2',
                max_lines=3,
                placeholder='Enter a negative prompt',
                visible=False,
            )
            with gr.Group():
                image = gr.Image(
                    label="Image", 
                    height=256, 
                    tool="editor", 
                    type="pil",
                    visible=False,
                )
                strength = gr.Slider(
                    label="Strength", 
                    minimum=0, 
                    maximum=1, 
                    step=0.01, 
                    value=0.55,
                    visible=False,
                )

            with gr.Row():
                aspect_ratio_selection = gr.Radio(
                    label='Aspect Ratios (W x H)', 
                    choices=list(ASPECT_RATIOS.keys()),
                    value='1024x1024',
                    visible=False,
                )
                
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
                with gr.Accordion(label='Conditioning Resolution', open=False):
                    with gr.Row():
                        original_width = gr.Slider(
                            label='Original Width',
                            minimum=1024,
                            maximum=4096,
                            step=32,
                            value=4096,
                        )
                        original_height = gr.Slider(
                            label='Original Height',
                            minimum=1024,
                            maximum=4096,
                            step=32,
                            value=4096,
                        )
                    with gr.Row():
                        target_width = gr.Slider(
                            label='Target Width',
                            minimum=1024,
                            maximum=4096,
                            step=32,
                            value=1024,
                            )
                        target_height = gr.Slider(
                            label='Target Height',
                            minimum=1024,
                            maximum=4096,
                            step=32,
                            value=1024,
                        )
                seed = gr.Slider(label='Seed',
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=0)
                
                randomize_seed = gr.Checkbox(label='Randomize seed', value=True)
                with gr.Row():
                    guidance_scale_base = gr.Slider(
                        label='Guidance scale',
                        minimum=1,
                        maximum=20,
                        step=0.1,
                        value=12.0)
                    num_inference_steps_base = gr.Slider(
                        label='Number of inference steps',
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50)

        with gr.Column(scale=2):
            with gr.Blocks():
                run_button = gr.Button('Generate')
            result = gr.Image(label='Result', show_label=False)

    gr.Examples(examples=examples,
                inputs=prompt,
                outputs=result,
                fn=generate,
                cache_examples=CACHE_EXAMPLES)
    
    # When use_prompt_2 is True
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

    # When init_image is True
    init_image.change(
        fn=lambda x: gr.update(visible=x),
        inputs=init_image,
        outputs=image,
        queue=False,
        api_name=False,
    )
    init_image.change(
        fn=lambda x: gr.update(visible=x),
        inputs=init_image,
        outputs=strength,
        queue=False,
        api_name=False,
    )
    init_image.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=init_image,
        outputs=aspect_ratio_presets,
        queue=False,
        api_name=False,
    )
    init_image.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=init_image,
        outputs=width,
        queue=False,
        api_name=False,
    )
    init_image.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=init_image,
        outputs=height,
        queue=False,
        api_name=False,
    )
    init_image.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=init_image,
        outputs=aspect_ratio_selection,
        queue=False,
        api_name=False,
    )

    # When init_image is set to False, update aspect_ratio_presets to True
    init_image.change(
        fn=lambda x: gr.update(value=True) if not x else gr.update(value=False),
        inputs=init_image,
        outputs=aspect_ratio_presets,
        queue=False,
        api_name=False,
    )

    # When aspect_ratio_presets is True
    aspect_ratio_presets.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=aspect_ratio_presets,
        outputs=width,
        queue=False,
        api_name=False,
    )
    aspect_ratio_presets.change(
        fn=lambda x: gr.update(visible=not x),
        inputs=aspect_ratio_presets,
        outputs=height,
        queue=False,
        api_name=False,
    )
    aspect_ratio_presets.change(
        fn=lambda x: gr.update(visible=x),
        inputs=aspect_ratio_presets,
        outputs=aspect_ratio_selection,
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
        guidance_scale_base,
        num_inference_steps_base,
        image,
        strength,
        init_image,
        aspect_ratio_presets,
        aspect_ratio_selection,
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
