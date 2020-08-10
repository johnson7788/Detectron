# Using Detectron

本文档提供了涵盖Detectron的简短教程，以对COCO数据集进行推理和训练。

- 有关Detectron的一般信息 please see [`README.md`](README.md).
- 有关安装说明, please see [`INSTALL.md`](INSTALL.md).

## 预训练模型推断

#### 1. Directory of Image Files
要在图像文件目录 (`demo/*.jpg` in this example)上进行推断, 
you can use the `infer_simple.py` tool. 
在此示例中，我们使用来自模型 model zoo ResNet-101-FPN backbone 端到端训练的Mask R-CNN模型

```
python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir output  \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo
```
Detectron应该从`--wts`参数指定的URL自动下载模型。
该工具将在“ --output-dir”指定的目录中以PDF格式输出检测结果的可视化。这是您应该看到的输出示例
 (for copyright information about the demo images see [`demo/NOTICE`](demo/NOTICE)).

<div align="center">
  <img src="demo/output/17790319373_bd19b24cfc_k_example_output.jpg" width="700px" />
  <p>Example Mask R-CNN output.</p>
</div>

**Notes:**

- 在您自己的高分辨率图像上进行推理时，Mask R-CNN可能会变慢，
这仅是因为要花费大量时间将预测的蒙版上采样到原始图像分辨率（尚未优化）。
如果`tools / infer_simple.py`报告的`misc_mask`时间过长（例如，远远超过20-90ms），
则可以诊断出此问题。解决方案是先调整图像大小，使短边在600-800px左右（确切的选择无关紧要），然后对调整后的图像进行推断。

#### 2. COCO Dataset
本示例说明如何使用单个GPU进行推理从model zoo运行端到端训练的Mask R-CNN模型。
As configured, this will run inference on all images in `coco_2014_minival` (which must be properly installed).

```
python tools/test_net.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    NUM_GPUS 1
```

Running inference with the same model using `$N` GPUs (e.g., `N=8`).

```
python tools/test_net.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    NUM_GPUS $N
```

在NVIDIA Tesla P100 GPU上，在此示例中，推理应该花费大约130-140毫秒/图像。


## 用Detectron训练模型

1788/5000
Character limit: 5000
这是一个很小的教程，显示了如何在COCO上训练模型。
该模型将是使用ResNet-50-FPN主干进行端到端训练的Faster R-CNN。
就本教程而言，我们将使用较短的训练时间表和较小的输入图像大小，以便训练和推理相对较快。
As a result, the box AP on COCO will be relatively low compared to our [baselines](MODEL_ZOO.md). 
提供此示例仅出于指导目的（即，不用于与出版物进行比较）。

#### 1. Training with 1 GPU

```
python tools/train_net.py \
    --cfg configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output
```

**Expected results:**

- Output (models, validation set detections, etc.) will be saved under `/tmp/detectron-output`
- On a Maxwell generation GPU (e.g., M40), training should take around 4.2 hours
- Inference time should be around 80ms / image (also on an M40)
- Box AP on `coco_2014_minival` should be around 22.1% (+/- 0.1% stdev measured over 3 runs)

### 2. Multi-GPU Training

We've also provided configs to illustrate training with 2, 4, and 8 GPUs using learning schedules that will be approximately equivalent to the one used with 1 GPU above. The configs are located at: `configs/getting_started/tutorial_{2,4,8}gpu_e2e_faster_rcnn_R-50-FPN.yaml`. For example, launching a training job with 2 GPUs will look like this:

```
python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/getting_started/tutorial_2gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /tmp/detectron-output
```

请注意，在训练结束后，我们还添加了--multi-gpu-testing标志，以指示Detectron并行化对多个GPU的推理（本示例中为2个；请参阅配置文件中的NUM_GPUS）

**Expected results:**

- Training should take around 2.3 hours (2 x M40)
- Inference time should be around 80ms / image (but in parallel on 2 GPUs, so half the total time)
- Box AP on `coco_2014_minival` should be around 22.1% (+/- 0.1% stdev measured over 3 runs)

要了解如何调整学习时间表（“线性缩放规则”），请研究这些教程配置文件并阅读我们的论文
[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
**除本教程外，我们所有已发布的配置均使用8个GPU。 如果您要使用少于8个GPU进行训练（或执行任何其他更改最小批量大小的操作），则必须了解如何根据线性缩放规则来操纵训练时间表。**

**Notes:**

- -此训练示例使用了相对较低的GPU计算模型，因此Caffe2 Python操作的开销相对较高。 
结果，随着GPU的数量从2增加到8，扩展性相对较差（例如，使用8个GPU进行训练大约需要0.9个小时，仅比使用1个GPU快4.5倍）。 
当使用更大，更多的GPU计算重型模型时，缩放比例会提高。
