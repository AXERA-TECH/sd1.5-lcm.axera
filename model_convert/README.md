# 模型转换

在 PC 上完成 ONNX 模型导出和 axmodel 模型编译. 注意, 如果想要获取工具链 `docker image` 需要走正规的发版流程.

## 安装依赖

```
git clone https://github.com/AXERA-TECH/sd1.5-lcm.axera.git
cd model_convert
pip install -r requirements.txt
```

## 导出模型（Huggingface -> ONNX）

下载 Huggingface 上对应的 Repo.

```sh
$ huggingface-cli download --resume-download latent-consistency/lcm-lora-sdv1-5 --local-dir latent-consistency/lcm-lora-sdv1-5

$ huggingface-cli download --resume-download Lykon/dreamshaper-7 --local-dir Lykon/dreamshaper-7
```

运行脚本 `sd15_export_onnx.py` 导出 `text_encoder`, `unet` 以及 `vae` 的 `onnx` 模型.

```sh
python3 sd15_export_onnx.py --input_path ./hugging_face/models/dreamshaper-7/ --input_lora_path ./hugging_face/models/lcm-lora-sdv1-5/ --output_path onnx-models
```

默认导出的 `vae_encoder` 模型输入图像尺寸为 `512x512`, 如果需要其它尺寸可以在命令行使用 `--isize 256x256` 这样的命令去修改导出的尺寸.

导出需要花费一定的时间, 请耐心等待.

导出后的文件目录如下所示:

```sh
✗ tree -L 1 onnx-models
onnx-models
├── a9a1a634-4cf5-11f0-b3ee-f5b7bf5aa809
├── sd15_text_encoder_sim.onnx
├── time_input_img2img.npy # 注意在不同任务时选用不同的 time 输入
├── time_input_txt2img.npy
├── unet.onnx
├── vae_decoder.onnx
└── vae_encoder.onnx

0 directories, 7 files
```

注意, 如果使用 **最新版** 工具链在编译模型时出现下面的错误:

```sh
Traceback (most recent call last):
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/graph_ir.py", line 999, in shapefn
    outputs_spec = self.impl.shapefn(inputs_tinfo)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/opset/oprdef.py", line 132, in shapefn
    outputs_shapes = self._shapefn(self._attrs, inputs_spec)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/operators/onnx/onnx_ops.py", line 1030, in <lambda>
    .setShapeInference(lambda attrs, inputs: onnx_shapefn_or_pyrun(attrs, inputs, True, "Reshape"))
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/operators/onnx/utils.py", line 159, in onnx_shapefn_or_pyrun
    model, inputs_data = make_one_node_model(attrs, inputs, outputs, op_type)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/operators/onnx/utils.py", line 79, in make_one_node_model
    assert v.is_const and v.data is not None
AssertionError
```

可以通过手动执行 `onnxslim` 命令, 通过优化导出的 `onnx` 模型来解决该错误. 示例命令如下:

```bash
pip3 install onnxslim
onnxslim vae_encoder.onnx vae_encoder_slim.onnx
```

## 生成量化数据集

运行脚本 `sd15_lora_prepare_data.py` 准备 `Pulsar2` 编译依赖的 `Calibration` 数据集

```sh
python3 sd15_lora_prepare_data.py --export_onnx_dir onnx-models[onnx导出目录]
```

代码执行结束后后进入 `datasets` 目录, 可以观察到目录结构如下所示:

```sh
datasets git:(yongqiang/dev) ✗ tree -L 1 calib_data_unet
calib_data_unet
├── data_0.npy
......
├── data_9.npy
└── data.tar
datasets git:(yongqiang/dev) ✗ tree -L 1 calib_data_vae
calib_data_vae
├── data_0_0.npy
......
├── data_9_3.npy
└── data.tar
```

## 模型转换

在 `Axera` 工具链 `docker` 中分别执行下面的命令进行模型编译.

(1) 编译 `sd15_text_encoder_sim.onnx` 模型

```sh
pulsar2 build --input onnx-models/sd15_text_encoder_sim.onnx  --output_dir axmodels  --output_name sd15_text_encoder_sim.axmodel --config configs/text_encoder_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```


(2) 编译 `vae_encoder.onnx` 模型

```sh
pulsar2 build --input onnx-models/vae_encoder.onnx  --output_dir axmodels  --output_name vae_encoder.axmodel --config configs/vae_encoder_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```


(3) 编译 `vae_decoder.onnx` 模型

```sh
pulsar2 build --input onnx-models/vae_decoder.onnx  --output_dir axmodels  --output_name vae_decoder.axmodel --config configs/vae_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```

(4) 编译 `unet.onnx` 模型

```sh
pulsar2 build  --input onnx-models/unet.onnx  --output_dir axmodels  --output_name unet.axmodel --config configs/unet_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```

