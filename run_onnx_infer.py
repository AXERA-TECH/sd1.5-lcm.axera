from typing import List, Union
import numpy as np
import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
# from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, AutoencoderKL
# from axengine import InferenceSession
import time
import argparse
from diffusers.utils import load_image
import PIL.Image
from typing import List, Optional, Tuple, Union
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import make_image_grid, load_image



########## Img2Img
PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

PipelineDepthInput = PipelineImageInput

# Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.IntTensor,
) -> torch.Tensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    # self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
    # Convert betas to alphas_bar_sqrt
    beta_start = 0.00085
    beta_end = 0.012
    num_train_timesteps = 1000
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def prepare_latents(image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                )

            init_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        init_latents = self.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # get latents
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents

    return latents

def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    r"""
    Convert a NumPy image to a PyTorch tensor.

    Args:
        images (`np.ndarray`):
            The NumPy image array to convert to PyTorch format.

    Returns:
        `torch.Tensor`:
            A PyTorch tensor representation of the images.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    r"""
    Convert a PIL image or a list of PIL images to NumPy arrays.

    Args:
        images (`PIL.Image.Image` or `List[PIL.Image.Image]`):
            The PIL image or list of images to convert to NumPy format.

    Returns:
        `np.ndarray`:
            A NumPy array representation of the images.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

def is_valid_image(image) -> bool:
    r"""
    Checks if the input is a valid image.

    A valid image can be:
    - A `PIL.Image.Image`.
    - A 2D or 3D `np.ndarray` or `torch.Tensor` (grayscale or color image).

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image to validate. It can be a PIL image, a NumPy array, or a torch tensor.

    Returns:
        `bool`:
            `True` if the input is a valid image, `False` otherwise.
    """
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)

def is_valid_image_imagelist(images):
    r"""
    Checks if the input is a valid image or list of images.

    The input can be one of the following formats:
    - A 4D tensor or numpy array (batch of images).
    - A valid single image: `PIL.Image.Image`, 2D `np.ndarray` or `torch.Tensor` (grayscale image), 3D `np.ndarray` or
      `torch.Tensor`.
    - A list of valid images.

    Args:
        images (`Union[np.ndarray, torch.Tensor, PIL.Image.Image, List]`):
            The image(s) to check. Can be a batch of images (4D tensor/array), a single image, or a list of valid
            images.

    Returns:
        `bool`:
            `True` if the input is valid, `False` otherwise.
    """
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False


def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Normalize an image array to [-1,1].

    Args:
        images (`np.ndarray` or `torch.Tensor`):
            The image array to normalize.

    Returns:
        `np.ndarray` or `torch.Tensor`:
            The normalized image array.
    """
    return 2.0 * images - 1.0

# Copy from: /home/baiyongqiang/miniforge-pypy3/envs/hf/lib/python3.9/site-packages/diffusers/image_processor.py#607
def preprocess(
    image: PipelineImageInput,
    height: Optional[int] = None,
    width: Optional[int] = None,
    resize_mode: str = "default",  # "default", "fill", "crop"
    crops_coords: Optional[Tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    """
    Preprocess the image input.

    Args:
        image (`PipelineImageInput`):
            The image input, accepted formats are PIL images, NumPy arrays, PyTorch tensors; Also accept list of
            supported formats.
        height (`int`, *optional*):
            The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
            height.
        width (`int`, *optional*):
            The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
        resize_mode (`str`, *optional*, defaults to `default`):
            The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
            the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
            resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
            center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
            image to fit within the specified width and height, maintaining the aspect ratio, and then center the
            image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
            supported for PIL image input.
        crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
            The crop coordinates for each image in the batch. If `None`, will not crop the image.

    Returns:
        `torch.Tensor`:
            The preprocessed image.
    """
    supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

    # # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
    # if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
    #     if isinstance(image, torch.Tensor):
    #         # if image is a pytorch tensor could have 2 possible shapes:
    #         #    1. batch x height x width: we should insert the channel dimension at position 1
    #         #    2. channel x height x width: we should insert batch dimension at position 0,
    #         #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
    #         #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
    #         image = image.unsqueeze(1)
    #     else:
    #         # if it is a numpy array, it could have 2 possible shapes:
    #         #   1. batch x height x width: insert channel dimension on last position
    #         #   2. height x width x channel: insert batch dimension on first position
    #         if image.shape[-1] == 1:
    #             image = np.expand_dims(image, axis=0)
    #         else:
    #             image = np.expand_dims(image, axis=-1)

    if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
        warnings.warn(
            "Passing `image` as a list of 4d np.ndarray is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
            FutureWarning,
        )
        image = np.concatenate(image, axis=0)
    if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
        warnings.warn(
            "Passing `image` as a list of 4d torch.Tensor is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
            FutureWarning,
        )
        image = torch.cat(image, axis=0)

    if not is_valid_image_imagelist(image):
        raise ValueError(
            f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
        )
    if not isinstance(image, list):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        if crops_coords is not None:
            image = [i.crop(crops_coords) for i in image]
        # if self.config.do_resize:
        #     height, width = self.get_default_height_width(image[0], height, width)
        #     image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
        # if self.config.do_convert_rgb:
        #     image = [self.convert_to_rgb(i) for i in image]
        # elif self.config.do_convert_grayscale:
        #     image = [self.convert_to_grayscale(i) for i in image]
        image = pil_to_numpy(image)  # to np
        image = numpy_to_pt(image)  # to pt

    elif isinstance(image[0], np.ndarray):
        image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

        # image = self.numpy_to_pt(image)

        # height, width = self.get_default_height_width(image, height, width)
        # if self.config.do_resize:
        #     image = self.resize(image, height, width)

    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        # if self.config.do_convert_grayscale and image.ndim == 3:
        #     image = image.unsqueeze(1)

        channel = image.shape[1]
        # don't need any preprocess if the image is latents
        # if channel == self.config.vae_latent_channels:
        #     return image

        # height, width = self.get_default_height_width(image, height, width)
        # if self.config.do_resize:
        #     image = self.resize(image, height, width)

    # expected range [0,1], normalize to [-1,1]
    do_normalize = True # self.config.do_normalize
    if do_normalize and image.min() < 0:
        warnings.warn(
            "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
            f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
            FutureWarning,
        )
        do_normalize = False
    if do_normalize:
        image = normalize(image)

    # if self.config.do_binarize:
    #     image = self.binarize(image)

    return image
##########


def get_args():
    parser = argparse.ArgumentParser(
        prog="StableDiffusion",
        description="Generate picture with the input prompt"
    )
    parser.add_argument("--prompt", "-p", type=str, required=False, default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k", help="the input text prompt")
    parser.add_argument("--text_model_dir", "-e", type=str, required=False, default="./models/", help="the dir of text encoder and tokenizer files")
    parser.add_argument("--unet_model", "-u", type=str, required=False, default="./models/unet.axmodel", help="the dir of unet.axmodel")
    parser.add_argument("--vae_encoder_model", type=str, required=False, default="./models/vae.axmodel", help="the dir of vae.axmodel")
    parser.add_argument("--vae_decoder_model", type=str, required=False, default="./models/vae.axmodel", help="the dir of vae.axmodel")
    parser.add_argument("--time_input", "-t", type=str, required=False, default="./models/time_input.npy", help="the dir of time input file")
    parser.add_argument("--save_dir", "-s", type=str, required=False, default="./lcm_lora_sdv1_5_axmodel.png", help="the save dir of the output image")
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
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir,
                                                 torch_dtype=torch.float32,
                                                 variant="fp16")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to("cpu"), attention_mask=None)

    prompt_embeds_npy = prompt_embeds[0].detach().numpy()
    return prompt_embeds_npy


def get_alphas_cumprod():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).detach().numpy()
    final_alphas_cumprod = alphas_cumprod[0]
    self_timesteps = np.arange(0, 1000)[::-1].copy().astype(np.int64)
    return alphas_cumprod, final_alphas_cumprod, self_timesteps


if __name__ == '__main__':

    """
    Usage:
        - python3 run_onnx_infer.py --prompt "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k" --unet_model output_onnx/modified_unet_sim_cut.onnx  --vae_encoder_model output_onnx/sd15_vae_encoder_sim.onnx --vae_decoder_model output_onnx/sd15_vae_decoder_sim.onnx  --time_input ./output_onnx/time_input.npy --save_dir ./img2img_output.png
    """
    args = get_args()
    prompt = args.prompt
    tokenizer_dir = args.text_model_dir + 'tokenizer'
    text_encoder_dir = args.text_model_dir + 'text_encoder'
    unet_model = args.unet_model
    vae_decoder_model = args.vae_decoder_model
    vae_encoder_model = args.vae_encoder_model
    time_input = args.time_input
    save_dir = args.save_dir

    print(f"prompt: {prompt}")
    print(f"text_tokenizer: {tokenizer_dir}")
    print(f"text_encoder: {text_encoder_dir}")
    print(f"unet_model: {unet_model}")
    print(f"vae_encoder_model: {vae_encoder_model}")
    print(f"vae_decoder_model: {vae_decoder_model}")
    print(f"time_input: {time_input}")
    print(f"save_dir: {save_dir}")

    # timesteps = np.array([999, 759, 499, 259]).astype(np.int64)

    # text encoder
    start = time.time()    
    # prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    # prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
    # prompt = "Caricature, a beautiful girl with black hair, 8k"
    prompt_embeds_npy = get_embeds(prompt, tokenizer_dir, text_encoder_dir)
    print(f"text encoder take {1000 * (time.time() - start)}ms")

    prompt_name = prompt.replace(" ", "_")
    latents_shape = [1, 4, 64, 64]
    # latent = torch.randn(latents_shape, generator=None, device="cpu", dtype=torch.float32,
    #                      layout=torch.strided).detach().numpy()

    alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()

    # load unet model and vae model
    start = time.time()
    vae_encoder = onnxruntime.InferenceSession(vae_encoder_model)
    unet_session_main = onnxruntime.InferenceSession(unet_model)
    vae_decoder = onnxruntime.InferenceSession(vae_decoder_model)
    print(f"load models take {1000 * (time.time() - start)}ms")

    # load time input file
    time_input = np.load(time_input)

    # load image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
    init_image = load_image(url) # U8, (512, 512, 3), RGB
    init_image_show = init_image
    
    # vae encoder inference
    vae_start = time.time()

    init_image = preprocess(init_image) # torch.Size([1, 3, 512, 512])
    if isinstance(init_image, torch.Tensor):
        init_image = init_image.detach().numpy() # 结果对齐

    vae_encoder_onnx_inp_name = vae_encoder.get_inputs()[0].name
    vae_encoder_onnx_out_name = vae_encoder.get_outputs()[0].name
    # vae_encoder_out.shape (1, 8, 64, 64), 数值基本对的上 (onnx fp32, torch fp16, 精度上会有点区别)
    # /home/baiyongqiang/miniforge-pypy3/envs/hf/lib/python3.9/site-packages/diffusers/models/autoencoders/autoencoder_kl.py#256
    vae_encoder_out = vae_encoder.run([vae_encoder_onnx_out_name], {vae_encoder_onnx_inp_name: init_image})[0] # encoder 输出: torch.Size([1, 8, 64, 64]), 最终结果 Size([1, 4, 64, 64])
    print(f"vae encoder inference take {1000 * (time.time() - vae_start)}ms")

    # vae_encoder 处理
    device = torch.device("cuda:0")
    vae_encoder_out = torch.from_numpy(vae_encoder_out).to(torch.float32)
    posterior = DiagonalGaussianDistribution(vae_encoder_out) # 数值基本对的上
    vae_encode_info = AutoencoderKLOutput(latent_dist=posterior)
    generator = torch.manual_seed(0)
    init_latents = retrieve_latents(vae_encode_info, generator=generator) # 数值基本对的上
    init_latents = init_latents * 0.18215 # 数值基本对的上
    init_latents = torch.cat([init_latents], dim=0)
    shape = init_latents.shape
    dtype = torch.float16
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # dtype 不同, 随机值不同
    # get latents
    timestep = torch.tensor([499]).to(device)
    init_latents = add_noise(init_latents.to(device), noise, timestep)
    latents = init_latents

    latents = latents.detach().cpu().numpy()
    latent = latents

    # unet inference loop
    unet_loop_start = time.time()
    timesteps = np.array([499, 259]).astype(np.int64)
    self_timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    step_index = [2, 3]
    for i, timestep in enumerate(timesteps):
        # print(i, timestep)
        
        unet_start = time.time()
        noise_pred = unet_session_main.run([unet_session_main.get_outputs()[0].name], {"sample": latent, \
                                            "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input[i], axis=0), \
                                            "encoder_hidden_states": prompt_embeds_npy})[0] # ['5771']
        # noise_pred = unet_session_main.run([unet_session_main.get_outputs()[0].name], {"sample": latent, \
        #                                     "t": np.array(timestep), \
        #                                     "encoder_hidden_states": prompt_embeds_npy})[0] # ['5771']

        sample = latent
        model_output = noise_pred

        # 1. get previous step value
        prev_step_index = step_index[i] + 1
        if prev_step_index < len(self_timesteps):
            prev_timestep = self_timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # if i < 3: # 3
        #     prev_timestep = timesteps[i + 1]
        # else:
        #     prev_timestep = timestep

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        scaled_timestep = timestep * 10
        c_skip = 0.5 ** 2 / (scaled_timestep ** 2 + 0.5 ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + 0.5 ** 2) ** 0.5
        predicted_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5) # 数值基本对齐

        denoised = c_out * predicted_original_sample + c_skip * sample
        if step_index[i] != 3:
            # noise = torch.randn(model_output.shape, generator=None, device="cpu", dtype=torch.float32,
            #                     layout=torch.strided).to("cpu").detach().numpy()
            device = torch.device("cpu")
            noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=torch.float16).numpy()
            prev_sample = (alpha_prod_t_prev ** 0.5) * denoised + (beta_prod_t_prev ** 0.5) * noise
        else:
            prev_sample = denoised

        latent = prev_sample

    print(f"unet loop take {1000 * (time.time() - unet_loop_start)}ms")

    # vae decoder inference
    vae_start = time.time()
    latent = latent / 0.18215
    image = vae_decoder.run([vae_decoder.get_outputs()[0].name], {"x": latent})[0] # ['784']
    print(f"vae decoder inference take {1000 * (time.time() - vae_start)}ms")

    # save result
    save_start = time.time() 
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])
    pil_image.save(save_dir)
    print(f"save image take {1000 * (time.time() - vae_start)}ms")

    from loguru import logger
    grid_img = make_image_grid([init_image_show, pil_image], rows=1, cols=2)
    grid_img.save(f"./lcm_lora_sdv1-5_imgGrid_output.png")
    logger.info(f"grid image saved in ./lcm_lora_sdv1-5_imgGrid_output.png")
