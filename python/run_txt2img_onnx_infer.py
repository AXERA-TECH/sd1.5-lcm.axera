from typing import List, Union
import numpy as np
import onnxruntime
# import axengine
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
import os
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="StableDiffusion",
        description="Generate picture with the input prompt"
    )
    parser.add_argument("--prompt", type=str, required=False, default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k", help="the input text prompt")
    parser.add_argument("--text_model_dir", type=str, required=False, default="./models/", help="Path to text encoder and tokenizer files")
    parser.add_argument("--unet_model", type=str, required=False, default="./models/unet.onnx", help="Path to unet ONNX model")
    parser.add_argument("--vae_decoder_model", type=str, required=False, default="./models/vae_decoder.onnx", help="Path to vae decoder ONNX model")
    parser.add_argument("--time_input", type=str, required=False, default="./models/time_input_txt2img.npy", help="Path to time input file")
    parser.add_argument("--save_dir", type=str, required=False, default="./txt2img_output_onnx.png", help="Path to the output image file")
    return parser.parse_args()

def maybe_convert_prompt(prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    if not isinstance(prompt, List):
        prompts = [prompt]
    else:
        prompts = prompt

    prompts = [_maybe_convert_prompt(p, tokenizer) for p in prompts]

    if not isinstance(prompt, List):
        return prompts[0]

    return prompts


def _maybe_convert_prompt(prompt: str, tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1

            prompt = prompt.replace(token, replacement)

    return prompt


def get_embeds(prompt = "Portrait of a pretty girl", tokenizer_dir = "./models/tokenizer", text_encoder_dir = "./models/text_encoder"):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    text_encoder = onnxruntime.InferenceSession(
        os.path.join(
            text_encoder_dir,
            "sd15_text_encoder_sim.onnx"
        ),
        providers=["CPUExecutionProvider"]
    )
    text_encoder_onnx_out = text_encoder.run(None, {"input_ids": text_input_ids.to("cpu").numpy()})[0]

    prompt_embeds_npy = text_encoder_onnx_out
    return prompt_embeds_npy


def get_alphas_cumprod():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).detach().numpy()
    final_alphas_cumprod = alphas_cumprod[0]
    self_timesteps = np.arange(0, 1000)[::-1].copy().astype(np.int64)
    return alphas_cumprod, final_alphas_cumprod, self_timesteps


if __name__ == '__main__':
    args = get_args()
    prompt = args.prompt
    tokenizer_dir = args.text_model_dir + 'tokenizer'
    text_encoder_dir = args.text_model_dir + 'text_encoder'
    unet_model = args.unet_model
    vae_decoder_model = args.vae_decoder_model
    time_input = args.time_input
    save_dir = args.save_dir

    print(f"prompt: {prompt}")
    print(f"text_tokenizer: {tokenizer_dir}")
    print(f"text_encoder: {text_encoder_dir}")
    print(f"unet_model: {unet_model}")
    print(f"vae_decoder_model: {vae_decoder_model}")
    print(f"time_input: {time_input}")
    print(f"save_dir: {save_dir}")

    timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    
    # text encoder
    start = time.time()    
    # prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    prompt_embeds_npy = get_embeds(prompt, tokenizer_dir, text_encoder_dir)
    print(f"text encoder take {(1000 * (time.time() - start)):.1f}ms")
    
    prompt_name = prompt.replace(" ", "_")
    latents_shape = [1, 4, 64, 64]
    latent = torch.randn(latents_shape, generator=None, device="cpu", dtype=torch.float32,
                         layout=torch.strided).detach().numpy()
    
    alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()
    
    # load unet model and vae model
    start = time.time()    
    unet_session_main = onnxruntime.InferenceSession(unet_model, providers=["CPUExecutionProvider"])
    vae_decoder = onnxruntime.InferenceSession(vae_decoder_model, providers=["CPUExecutionProvider"])
    print(f"load models take {(1000 * (time.time() - start)):.1f}ms")
    
    # load time input file
    time_input = np.load(time_input)
    
    # unet inference loop
    unet_loop_start = time.time()    
    for i, timestep in enumerate(timesteps):
        # print(i, timestep)
        
        unet_start = time.time()
        noise_pred = unet_session_main.run(None, {"sample": latent, \
                                            "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input[i], axis=0), \
                                            "encoder_hidden_states": prompt_embeds_npy})[0]
                                            
        print(f"unet once take {(1000 * (time.time() - unet_start)):.1f}ms")

        sample = latent
        model_output = noise_pred
        if i < 3:
            prev_timestep = timesteps[i + 1]
        else:
            prev_timestep = timestep

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        scaled_timestep = timestep * 10
        c_skip = 0.5 ** 2 / (scaled_timestep ** 2 + 0.5 ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + 0.5 ** 2) ** 0.5
        predicted_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        denoised = c_out * predicted_original_sample + c_skip * sample

        if i != 3:
            noise = torch.randn(model_output.shape, generator=None, device="cpu", dtype=torch.float32,
                                layout=torch.strided).to("cpu").detach().numpy()
            prev_sample = (alpha_prod_t_prev ** 0.5) * denoised + (beta_prod_t_prev ** 0.5) * noise
        else:
            prev_sample = denoised

        latent = prev_sample

    print(f"unet loop take {(1000 * (time.time() - unet_loop_start)):.1f}ms")

    # vae inference
    vae_start = time.time()    
    latent = latent / 0.18215
    image = vae_decoder.run(None, {"x": latent})[0]
    print(f"vae inference take {(1000 * (time.time() - vae_start)):.1f}ms")
    
    # save result
    save_start = time.time() 
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])
    pil_image.save(save_dir)
    print(f"save image take {(1000 * (time.time() - save_start)):.1f}ms")
