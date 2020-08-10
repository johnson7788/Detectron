# 安装Detectron

本文档介绍如何安装Detectron，其依赖项（包括Caffe2）和COCO数据集。

- 有关Detectron的一般信息, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, 各种标准Python软件包和COCO API；有关安装这些依赖项的说明，请参见下文

**Notes:**

- Detectron operators 当前没有CPU实现；需要GPU系统。
- Detectron 已通过CUDA 8.0和cuDNN 6.0.21进行了广泛的测试。

## Caffe2

要安装具有CUDA支持的Caffe2，请遵循[Caffe2网站](https://caffe2.ai/)上的
[安装说明](https://caffe2.ai/docs/getting-started.html)。 **如果已经安装了Caffe2，请确保将Caffe2更新到包含[Detectron模块]

请先运行以下命令并按照注释中的指示检查其输出，然后再继续操作，以确保您的Caffe2安装成功。

```
# 检查Caffe2构建是否成功
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# 检查Caffe2 GPU构建是否成功
# This must print a number > 0 in order to use Detectron
python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

如果找不到`caffe2` Python软件包，则可能需要调整`PYTHONPATH`环境变量以包含其位置
 (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

## Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
COCOAPI=/content/cocoapi
mkdir -p $COCOAPI
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

## Detectron 安装

Clone the Detectron repository:

```
%%bash
DETECTRON=/content/detectron
mkdir -p $DETECTRON
git clone https://github.com/facebookresearch/detectron $DETECTRON
```

Install Python dependencies:

```
pip install -r $DETECTRON/requirements.txt
```

Set up Python modules:

```
cd $DETECTRON && make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](detectron/tests/test_spatial_narrow_as_op.py)):

```
# 检查detectron是否安装正常
python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py
```

## That's All You Need for Inference

此时，您可以使用预训练的Detectron模型进行推理。看看我们的[inference tutorial](GETTING_STARTED.md)为例。如果要在COCO数据集上训练模型，请继续执行安装说明。

## Datasets

Detectron通过符号链接 `detectron/datasets/data` to the actual locations where the dataset images and annotations are stored.
有关如何为COCO和其他数据集创建符号链接的说明，, please see [`detectron/datasets/data/README.md`](detectron/datasets/data/README.md).

创建符号链接后，这就是开始训练模型所需的全部。

## Advanced Topic: Custom Operators for New Research Projects

Please read the custom operators section of the [`FAQ`](FAQ.md) first.

为了方便起见，我们为构建自定义operators提供CMake支持。所有自定义运算符都内置在单个库中，该库可以从Python动态加载。

Place your custom operator implementation under [`detectron/ops/`](detectron/ops/) and see [`detectron/tests/test_zero_even_op.py`](detectron/tests/test_zero_even_op.py) for an example of how to load custom operators from Python.


构建自定义运算符库：

```
cd $DETECTRON && make ops
```

Check that the custom operator tests pass:

```
python $DETECTRON/detectron/tests/test_zero_even_op.py
```

## Docker Image

我们提供了一个 [`Dockerfile`](docker/Dockerfile) 
可用于在满足顶部概述要求的Caffe2图像之上构建一个Detectron图像。 
如果您想使用不同于我们默认使用的Caffe2图像，请确保它包含[Detectron module](https://github.com/pytorch/pytorch/tree/master/modules/detectron).

Build the image:

```
cd $DETECTRON/docker
docker build -t detectron:c2-cuda9-cudnn7 .
```

Run the image (e.g. for [`BatchPermutationOp test`](detectron/tests/test_batch_permutation_op.py)):

```
nvidia-docker run --rm -it detectron:c2-cuda9-cudnn7 python detectron/tests/test_batch_permutation_op.py
```

## Troubleshooting

如果发生Caffe2安装问题，请先阅读相关Caffe2  [installation instructions](https://caffe2.ai/docs/getting-started.html) first. 
在下文中，我们提供了有关Caffe2和Detectron的其他疑难解答提示。


### Caffe2 Operator Profiling

Caffe2 comes with performance [`profiling`](https://github.com/pytorch/pytorch/tree/master/caffe2/contrib/prof)
support which you may find useful for benchmarking or debugging your operators
(see [`BatchPermutationOp test`](detectron/tests/test_batch_permutation_op.py) for example usage).
Profiling support is not built by default and you can enable it by setting
the `-DUSE_PROF=ON` flag when running Caffe2 CMake.

### CMake Cannot Find CUDA and cuDNN

Sometimes CMake has trouble with finding CUDA and cuDNN dirs on your machine.

When building Caffe2, you can point CMake to CUDA and cuDNN dirs by running:

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/dir \
  -DCUDNN_ROOT_DIR=/path/to/cudnn/root/dir
```

Similarly, when building custom Detectron operators you can use:

```
cd $DETECTRON
mkdir -p build && cd build
cmake .. \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/dir \
  -DCUDNN_ROOT_DIR=/path/to/cudnn/root/dir
make
```

请注意，您可以使用相同的命令使CMake在计算机上安装的多个版本中使用CUDA和cuDNN的特定版本。

### Protobuf Errors

Caffe2使用protobuf作为其序列化格式，并且需要版本“ 3.2.0”或更高版本。
如果您的protobuf版本较旧，则可以从Caffe2 protobuf子模块构建protobuf并改用该版本。

要构建Caffe2 protobuf子模块：

```
# CAFFE2=/path/to/caffe2
cd $CAFFE2/third_party/protobuf/cmake
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/c2_tp_protobuf \
  -Dprotobuf_BUILD_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="-fPIC"
make install
```

To point Caffe2 CMake to the newly built protobuf:

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DPROTOBUF_PROTOC_EXECUTABLE=$HOME/c2_tp_protobuf/bin/protoc \
  -DPROTOBUF_INCLUDE_DIR=$HOME/c2_tp_protobuf/include \
  -DPROTOBUF_LIBRARY=$HOME/c2_tp_protobuf/lib64/libprotobuf.a
```

如果同时安装了系统软件包和anaconda软件包，则protobuf可能也会遇到问题。
由于版本可能在编译时或运行时混合在一起，因此可能会导致问题。
通过遵循上面的命令，也可以解决此问题。

### Caffe2 Python Binaries

如果您遇到CMake在以下情况下找不到所需的Python路径的问题：
构建Caffe2 Python二进制文件（例如在virtualenv中），您可以尝试将Caffe2 CMake指向python
库并通过以下方式包含目录：

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DPYTHON_LIBRARY=$(python -c "from distutils import sysconfig; print(sysconfig.get_python_lib())") \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")
```

### Caffe2 with NNPACK Build

Detectron does not require Caffe2 built with NNPACK support. If you face NNPACK related issues during Caffe2 installation, you can safely disable NNPACK by setting the `-DUSE_NNPACK=OFF` CMake flag.

### Caffe2 with OpenCV Build

Analogously to the NNPACK case above, you can disable OpenCV by setting the `-DUSE_OPENCV=OFF` CMake flag.

### COCO API Undefined Symbol Error

If you encounter a COCO API import error due to an undefined symbol, as reported [here](https://github.com/cocodataset/cocoapi/issues/35),
make sure that your python versions are not getting mixed. For instance, this issue may arise if you have
[both system and conda numpy installed](https://stackoverflow.com/questions/36190757/numpy-undefined-symbol-pyfpe-jbuf).

### CMake Cannot Find Caffe2

如果您在构建自定义运算符时遇到CMake无法找到Caffe2软件包的问题，
make sure you have run `make install` as part of your Caffe2 installation process.
