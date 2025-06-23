# SD1.5-LCM.Axera

基于 StableDiffusion 1.5 LCM 项目, 展示该项目在 `Axera` 芯片上部署的流程.

支持芯片:

- AX650N

原始模型请参考:

- [Latent Consistency Model (LCM) LoRA: SDv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [Dreamshaper 7](https://huggingface.co/Lykon/dreamshaper-7)

克隆仓库:

```sh
git clone https://github.com/AXERA-TECH/sd1.5-lcm.axera
cd sd1.5-lcm.axera
```

其中 `model_convert` 文件夹用于模型编译转换, 具体请参考[模型转换文档](model_convert/README.md). `python` 文件夹内主要存放运行脚本.
 
## ONNX 以及 AXMODEL 推理

当模型文件编译转换完成之后, 可以进入 `python` 文件夹内执行生图任务.

```sh
cd sd1.5-lcm.axera/python
```

文件夹内可以按照下面的格式进行组织

```sh
ai@ai-bj ~/yongqiang/sd1.5-lcm.axera/python $ tree -L 2 models/
models/
├── 7ffcf62c-d292-11ef-bb2a-9d527016cd35
├── text_encoder
│   ├── config.json
│   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── sd15_text_encoder_sim.axmodel
│   └── sd15_text_encoder_sim.onnx
├── time_input_img2img.npy
├── time_input_txt2img.npy
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet.axmodel
├── vae_decoder.axmodel
└── vae_encoder.axmodel

2 directories, 15 files
```

这样可以很方便地以下面的代码执行 onnx 或 axmodel 的文生图和图生图任务.

```sh
python3 run_txt2img_axe[onnx]_infer.py --prompt "your prompt"
python3 run_img2img_axe[onnx]_infer.py --prompt "your prompt"
```

### 运行

在 `Axera` 开发板上推理时会用到前面编译好的 `axmodel` 模型, 推理脚本为 `run_txt2img_axe_infer.py` 和 `run_img2img_axe_infer.py`. 一个用于文生图任务, 一个用于图生图任务.

#### 文生图任务

**Input Prompt**

```
"((masterpiece,best quality))1 young beautiful girl,ultra detailed,official art,unity 8k wallpaper,masterpiece, best quality, official art, extremely detailed CG unity 8k wallpaper, highly detailed, 1 girl, aqua eyes, light smile, ((grey hair)), hair flower, bracelet, choker, ribbon, JK, look at viewer, on the beach, in summer,"
```

**Output Log**
```sh
ai@ai-bj ~/yongqiang/sd1.5-lcm.axera/python $ python3 run_txt2img_axe_infer.py --prompt "((masterpiece,best quality))1 young beautiful girl,ultra detailed,official art,unity 8k wallpaper,masterpiece, best quality, official art, extremely detailed CG unity 8k wallpaper, highly detailed, 1 girl, aqua eyes, light smile, ((grey hair)), hair flower, bracelet, choker, ribbon, JK, look at viewer, on the beach, in summer,"
[INFO] Available providers:  ['AXCLRTExecutionProvider']
prompt: ((masterpiece,best quality))1 young beautiful girl,ultra detailed,official art,unity 8k wallpaper,masterpiece, best quality, official art, extremely detailed CG unity 8k wallpaper, highly detailed, 1 girl, aqua eyes, light smile, ((grey hair)), hair flower, bracelet, choker, ribbon, JK, look at viewer, on the beach, in summer,
text_tokenizer: ./models/tokenizer
text_encoder: ./models/text_encoder
unet_model: ./models/unet.axmodel
vae_decoder_model: ./models/vae_decoder.axmodel
time_input: ./models/time_input_txt2img.npy
save_dir: ./txt2img_output_axe.png
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.4 9215b7e5
text encoder axmodel take 9.8ms
get_embeds take 11.5ms
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.3 972f38ca
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.3 972f38ca
load models take 15280.1ms
unet once take 436.0ms
unet once take 437.8ms
unet once take 437.5ms
unet once take 437.8ms
unet loop take 1753.4ms
vae inference take 930.4ms
save image take 123.3ms
```

**Output Image**

![](assets/txt2img_output_axe.png)

#### 图生图任务

**Input Image & Prompt**

输入一张初始图像以及对应的 prompt, 图像如下:

![](assets/img2img-init.png)

运行

```
python3 run_img2img_axe_infer.py --init_image models/img2img-init.png --prompt "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
```

**Output Image**

![](assets/lcm_lora_sdv1-5_imgGrid_output.png)

图中(右)即为图生图的结果.

## 相关项目

NPU 工具链 [Pulsar2 在线文档](https://pulsar2-docs.readthedocs.io/zh-cn/latest/)

## 技术讨论
Github issues
QQ 群: 139953715

## 免责声明

- 本项目只用于指导如何将 [Latent Consistency Model (LCM) LoRA: SDv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) 开源项目的模型部署在 AX650N 上
- 该模型存在的固有的局限性, 可能产生错误的、有害的、冒犯性的或其他不良的输出等内容与 AX650N 以及本仓库所有者无关
- [免责声明](./Disclaimer.md)
