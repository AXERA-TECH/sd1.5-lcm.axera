import argparse
import pathlib
import numpy as np
import onnx
import onnxsim
import torch
import os
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
from loguru import logger


"""
test env:
    protobuf:3.20.3
    onnx:1.16.0
    onnxsim:0.4.36
    torch:2.1.2+cu121
    transformers:4.45.0
"""


def extract_by_hand(input_model):
    input_graph = input_model
    to_remove_node = []
    for node in input_graph.node:
        if (
            node.name.startswith("/time_proj")
            or node.name.startswith("/time_embedding")
            or node.name
            in [
                "/down_blocks.0/resnets.0/act_1/Sigmoid",
                "/down_blocks.0/resnets.0/act_1/Mul",
            ]
        ):
            to_remove_node.append(node)
        else:
            pass
    for node in to_remove_node:
        input_graph.node.remove(node)
    to_remove_input = []
    for input in input_graph.input:
        if input.name in ["t"]:
            to_remove_input.append(input)
    for input in to_remove_input:
        input_graph.input.remove(input)
    new_input = []
    for value_info in input_graph.value_info:
        if value_info.name == "/down_blocks.0/resnets.0/act_1/Mul_output_0":
            new_input.append(value_info)
    input_graph.input.extend(new_input)


def extract_unet(args):

    input_path = args.input_path
    input_lora_path = args.input_lora_path
    output_path = args.output_path
    is_img2img = args.img2img
    isize = args.isize
    vae_encoder_img_h, vae_encoder_img_w = list(map(int, isize.split("x")))
    unet_feat_h = vae_encoder_img_h // 8
    unet_feat_w = vae_encoder_img_w // 8

    pipe = AutoPipelineForText2Image.from_pretrained(
        input_path, torch_dtype=torch.float32, variant="fp16"
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # load and fuse lcm lora
    pipe.load_lora_weights(
        str(pathlib.Path(input_lora_path) / "pytorch_lora_weights.safetensors")
    )
    pipe.fuse_lora()
    """
        extract unet
    """
    extract_unet = True
    if extract_unet:
        pipe.unet.eval()

        class UNETWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, sample=None, t=None, encoder_hidden_states=None):
                return self.unet.forward(sample, t, encoder_hidden_states)

        example_input = {
            "sample": torch.rand([1, 4, unet_feat_h, unet_feat_w], dtype=torch.float32),
            "t": torch.from_numpy(np.array([1], dtype=np.int64)),
            "encoder_hidden_states": torch.rand([1, 77, 768], dtype=torch.float32),
        }

        unet_path = pathlib.Path(output_path) / "unet"
        if not unet_path.exists():
            unet_path.mkdir()

        unet_onnx_save_path = str(unet_path / "unet.onnx")
        torch.onnx.export(
            UNETWrapper(pipe.unet),
            tuple(example_input.values()),
            unet_onnx_save_path,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            input_names=list(example_input.keys()),
        )
        unet = onnx.load(unet_onnx_save_path)
        unet_sim, check = onnxsim.simplify(unet)
        assert check, "Simplified ONNX model could not be validated"

        extract_by_hand(unet_sim.graph)
        # onnx.checker.check_model(unet_sim)
        onnx.save(
            unet_sim,
            unet_onnx_save_path,
            save_as_external_data=True,
        )

        import shutil
        shutil.copy2(unet_onnx_save_path, pathlib.Path(output_path))
        # shutil.copy2(pathlib.Path(output_path) + "/", pathlib.Path(output_path))

        import uuid
        uuid_file = None
        for file in unet_path.iterdir():
            if file.is_file():
                try:
                    # 验证文件名是否为有效的UUID格式
                    uuid.UUID(file.stem)  # 使用stem去掉扩展名
                    uuid_file = file
                    break  # 找到第一个UUID文件就退出
                except ValueError:
                    continue
        if uuid_file is None:
            raise FileNotFoundError(f"在 {unet_path} 中未找到UUID格式的文件")
        
        # import pdb; pdb.set_trace()
        shutil.copy2(uuid_file, pathlib.Path(output_path))
        shutil.rmtree(unet_path)

        logger.info(f"The unet_sim was successfully saved in {unet_onnx_save_path}/unet.onnx")


    """
        precompute time embedding
    """
    time_input = np.zeros([4, 1280], dtype=np.float32)
    # timesteps = np.array([499, 259]).astype(np.int64) if is_img2img \
    #     else np.array([999, 759, 499, 259]).astype(np.int64)

    timesteps_dict = {
        "time_input_img2img": np.array([499, 259]).astype(np.int64),
        "time_input_txt2img": np.array([999, 759, 499, 259]).astype(np.int64),
    }

    for key, timesteps in timesteps_dict.items():
        for i, t in enumerate(timesteps):
            tt = torch.from_numpy(np.array([t])).to(torch.float32)
            sample = pipe.unet.time_proj(tt)
            res = pipe.unet.time_embedding(sample)
            res = torch.nn.functional.silu(res)
            res_npy = res.detach().numpy()[0]
            time_input[i, :] = res_npy
        np.save(str(pathlib.Path(output_path) / f"{key}.npy"), time_input)


def extract_vae(args):

    input_path = args.input_path
    output_path = args.output_path
    isize = args.isize

    vae_encoder_img_h, vae_encoder_img_w = list(map(int, isize.split("x")))
    vae_decoder_img_h = vae_encoder_img_h // 8
    vae_decoder_img_w = vae_encoder_img_w // 8

    vae = AutoencoderKL.from_pretrained(
        str(pathlib.Path(input_path) / "vae"), torch_dtype=torch.float32, variant="fp16"
    )
    vae.eval()

    # 导出 VAE 的 Decoder 部分
    dummy_input = torch.rand([1, 4, vae_decoder_img_h, vae_decoder_img_w], dtype=torch.float32)
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, conv_quant, decoder):
            super().__init__()
            self.conv_quant = conv_quant
            self.decoder = decoder

        def forward(self, sample=None):
            sample = self.conv_quant(sample)
            decoder = self.decoder(sample)
            return decoder

    vae_decoder_wrapper = VAEDecoderWrapper(vae.post_quant_conv, vae.decoder)
    vae_decoder_save_path = str(pathlib.Path(output_path) / "vae_decoder.onnx")

    torch.onnx.export(
        vae_decoder_wrapper,
        dummy_input,
        vae_decoder_save_path,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        input_names=["x"],
    )
    vae_decoder_onnx = onnx.load(vae_decoder_save_path)
    vae_decoder_onnx_sim, _ = onnxsim.simplify(vae_decoder_onnx)
    onnx.save(vae_decoder_onnx_sim, vae_decoder_save_path)
    logger.info(f"The vae_decoder onnx model was successfully saved in {vae_decoder_save_path}")

    # 导出 VAE 的 Encoder 部分
    dummy_input = torch.rand([1, 3, vae_encoder_img_h, vae_encoder_img_w], dtype=torch.float32)
    dynamic_axes = {
        'image_sample': {2: 'H', 3: 'W'},
    }
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder, pre_quant_conv):
            super().__init__()
            self.encoder = encoder
            self.pre_quant_conv = pre_quant_conv

        def forward(self, sample=None):
            sample = self.encoder(sample)
            latent_sample = self.pre_quant_conv(sample)
            return latent_sample

    vae_encoder_wrapper = VAEEncoderWrapper(vae.encoder, vae.quant_conv)
    vae_encoder_save_path = str(pathlib.Path(output_path) / "vae_encoder.onnx")

    torch.onnx.export(
        vae_encoder_wrapper,
        dummy_input,
        vae_encoder_save_path,
        opset_version=17,
        verbose=False,
        do_constant_folding=True,
        input_names=["image_sample"],
        output_names=["latent_sample"],
        # dynamic_axes=dynamic_axes,
    )
    vae_encoder_onnx = onnx.load(vae_encoder_save_path)
    vae_encoder_onnx_sim, _ = onnxsim.simplify(vae_encoder_onnx)
    onnx.save(vae_encoder_onnx_sim, vae_encoder_save_path)
    logger.info(f"The vae_encoder onnx model was successfully saved in {vae_encoder_save_path}")


def extract_text_encoder(args):
    input_path = args.input_path
    output_path = args.output_path
    max_length = 77

    text_encoder = CLIPTextModel.from_pretrained(
        str(pathlib.Path(input_path) / "text_encoder"), torch_dtype=torch.float32, variant="fp16"
    )

    text_encoder.eval()

    dummy_input = torch.randint(1, 100, (1, max_length), dtype=torch.int64)
    text_encoder_save_path = str(pathlib.Path(output_path) / "sd15_text_encoder_sim.onnx")

    torch.onnx.export(
        text_encoder.text_model,
        dummy_input,
        text_encoder_save_path,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
    )
    text_encoder_onnx = onnx.load(text_encoder_save_path)
    text_encoder_onnx_sim, _ = onnxsim.simplify(text_encoder_onnx)
    extract_text_encoder_by_hand(text_encoder_onnx_sim.graph)
    onnx.save(text_encoder_onnx_sim, text_encoder_save_path)
    logger.info(f"The text_encoder was successfully saved in {text_encoder_save_path}")


def extract_text_encoder_by_hand(input_model):
    input_graph = input_model
    to_remove_node_name = ["/Cast", "/ArgMax", "/Add", "/Flatten", "/Gather_2", "/Reshape_1", "/Reshape_2"]
    to_remove_node = []
    for node in input_graph.node:
        if node.name in to_remove_node_name:
            to_remove_node.append(node)
        else:
            pass
    for node in to_remove_node:
        input_graph.node.remove(node)

    to_remove_output_name = ["1853"]
    to_remove_output = []
    for output in input_graph.output:
        if output.name in to_remove_output_name:
            to_remove_output.append(output)
    for output in to_remove_output:
        input_graph.output.remove(output)


if __name__ == "__main__":

    """
    Usage:
        python3 model_convert/sd15_export_onnx.py \
        --input_path ../hugging_face/models/dreamshaper-7 \
        --input_lora_path ../hugging_face/models/lcm-lora-sdv1-5 \
        --output_path onnx-models [--isize 256x256]
    """
    parser = argparse.ArgumentParser(description="unet extract")
    parser.add_argument("--input_path", required=True, help="download sd_15 path")
    parser.add_argument("--input_lora_path", help="download lora weight path", required=True)
    parser.add_argument("--output_path", help="output path", required=True)
    parser.add_argument("--img2img", action="store_true", help="Deprecated: support image-to-image mode")
    parser.add_argument("--isize", default="512x512", help="vae encoder input image size")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    logger.info("Start the model transformation ...")

    extract_text_encoder(args)
    extract_unet(args)
    extract_vae(args)

