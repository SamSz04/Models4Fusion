import os
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.linen.pooling import max_pool


# ==========================================
# 1. 设置 XLA Dump 环境变量
# ==========================================
# 这告诉编译器把中间表示（IR）保存到哪里
# --xla_dump_hlo_as_text: 保存为 .txt 文件，方便阅读
# --xla_dump_hlo_as_proto: 保存为 .pb 文件，方便工具解析
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./xla_dumps_resnet "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_dot"
)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    block_layers: list[int]
    num_classes: int

    def resnet50(num_classes: int = 1000):
        return ModelConfig([3, 4, 6, 3], num_classes=num_classes)

    def resnet152(num_classes: int = 1000):
        return ModelConfig([3, 8, 36, 3], num_classes=num_classes)


class Bottleneck(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), strides=1, padding=0, use_bias=False, rngs=rngs
        )
        self.bn0 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

        self.conv1 = nnx.Conv(
            out_channels, out_channels, kernel_size=(3, 3), strides=stride, padding=1, use_bias=False, rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

        self.conv2 = nnx.Conv(
            out_channels, out_channels * 4, kernel_size=(1, 1), strides=1, padding=0, use_bias=False, rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(out_channels * 4, use_running_average=True, rngs=rngs)

        self.downsample = downsample

    def __call__(self, x):
        identity = x

        x = self.conv0(x)
        x = self.bn0(x)
        x = nnx.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return nnx.relu(x + identity)


class Downsample(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), strides=stride, padding=0, use_bias=False, rngs=rngs
        )
        self.bn = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        return self.bn(x)


class BlockGroup(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks, stride: int, *, rngs: nnx.Rngs):
        self.blocks = nnx.List()

        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = Downsample(in_channels, out_channels * 4, stride, rngs=rngs)

        self.blocks.append(Bottleneck(in_channels, out_channels, stride, downsample, rngs=rngs))
        for _ in range(1, blocks):
            self.blocks.append(Bottleneck(out_channels * 4, out_channels, stride=1, downsample=None, rngs=rngs))

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Stem(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(3, 64, kernel_size=(7, 7), strides=2, padding=3, use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(64, use_running_average=True, rngs=rngs)
        self.pool = partial(max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nnx.relu(x)
        x = self.pool(x)
        return x


class ResNet(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.stem = Stem(rngs=rngs)

        self.layer0 = BlockGroup(64, 64, cfg.block_layers[0], stride=1, rngs=rngs)
        self.layer1 = BlockGroup(256, 128, cfg.block_layers[1], stride=2, rngs=rngs)
        self.layer2 = BlockGroup(512, 256, cfg.block_layers[2], stride=2, rngs=rngs)
        self.layer3 = BlockGroup(1024, 512, cfg.block_layers[3], stride=2, rngs=rngs)

        self.pool = partial(lambda x: x.mean(axis=(1, 2)))
        self.fc = nnx.Linear(2048, cfg.num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.stem(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(x)


@jax.jit
def forward(model, x):
    return model(x)


# ==========================================
# 2. 执行逻辑：初始化与触发编译
# ==========================================
if __name__ == "__main__":
    print("🚀 正在初始化 ResNet50 (NNX)...")

    # 1. 初始化模型
    # NNX 的特点是模型对象本身持有参数 (Stateful)
    rngs = nnx.Rngs(0)  # 设置随机种子
    config = ModelConfig.resnet50()
    model = ResNet(config, rngs=rngs)

    # 2. 构建 Dummy Input (伪造输入)
    # ResNet 标准输入通常是 (Batch, Height, Width, Channels)
    # 这里使用 Batch=1 来获取单次推理的图
    input_shape = (1, 224, 224, 3)
    x = jnp.ones(input_shape, dtype=jnp.float32)


    # 3. 定义 JIT 编译的推理函数
    # NNX 模型可以直接作为参数传递给 JIT 函数，JAX 会自动处理其 PyTree 结构
    @jax.jit
    def inference_step(model, x):
        # 注意：你的 BatchNorm 设置了 use_running_average=True
        # 这意味着这是纯推理模式，不会更新 Batch Stats，非常适合 Dump 静态图
        return model(x)


    print("⚡️ 开始 JIT 编译并触发 XLA Dump...")
    print(f"   输入形状: {input_shape}")

    # 4. 运行一次以触发 Tracing 和编译
    # 这一步完成后，./xla_dumps_resnet 文件夹下就会生成 HLO 文件
    logits = inference_step(model, x)

    print(f"✅ 完成！输出 Logits 形状: {logits.shape}")
    print("📁 请查看当前目录下的 'xla_dumps_resnet' 文件夹获取 HLO 文件。")
    print("   重点寻找包含 'before_optimizations' 和 'ir_with_opt' 的 txt 文件。")