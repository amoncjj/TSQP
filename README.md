# TSQP：基于 Gramine 的 LLaMA 3.2 TEE 推理框架

本项目演示如何将 Meta LLaMA 3.2 1B 模型部署在可信执行环境（TEE）中运行，同时将线性与嵌入层卸载到 GPU，以获得兼顾安全性与性能的推理方案。除此之外，项目还提供纯 TEE 推理流程，用于对比两种方式的性能差异。

## 目录结构一览

```text
TSQP/
├── tee_gpu/                # TEE + GPU 协同推理实现
│   ├── server.py           # GPU 侧 gRPC 服务端，托管线性/嵌入层
│   ├── tee_runner.py       # TEE 内客户端，负责非线性算子与生成流程
│   ├── msg.proto           # gRPC 协议定义，生成 msg_pb2*.py
│   ├── Makefile            # gRPC 代码生成 + Gramine 构建
│   ├── pytorch.manifest.template
│   ├── tee_runner.manifest.template
│   └── benchmarks/
│       └── run_split_benchmark.sh
├── tee_only_llama/         # 全部算子在 TEE 内执行的实现
│   ├── tee_runner.py
│   └── benchmarks/
│       ├── run_full_tee_benchmark.sh
│       └── compare_split_vs_tee.sh
├── requirements.txt        # Python 依赖（请在目标环境安装）
└── README.md               # 项目说明（当前文件）
```

## 核心组件说明

- **tee_gpu/server.py**：加载 LLaMA 模型，将所有 `nn.Linear` 与 `nn.Embedding` 模块注册为远程模块，通过 gRPC 对外暴露推理接口。
- **tee_gpu/tee_runner.py**：在 TEE 中运行的推理脚本，负责：
  1. 启动时向 GPU 端注册所需的线性模块；
  2. 同步非线性参数/缓冲区；
  3. 在生成过程中，将线性层调用转发给 GPU 服务，非线性部分在 TEE 内本地执行。
- **tee_only_llama/tee_runner.py**：完全在 TEE 内部加载并运行 LLaMA，可作为安全性基线。
- **Manifest 模板**：`pytorch.manifest.template`、`tee_runner.manifest.template` 用于生成 Gramine manifest，控制文件白名单、环境变量等。

## 运行前的准备

1. **准备运行环境**
   - 建议使用 Python 3.10+。
   - 目标平台需具备 Intel SGX（运行 Gramine-SGX）与可用 GPU（split 模式）
   - 在目标平台按需安装 `requirements.txt` 中依赖（包含 `grpcio`, `torch`, `transformers` 等）。

2. **获取模型权重**
   - 将已授权下载的 `Llama-3.2-1B-Instruct` 权重放在 `weights/llama3.2-1b/` 目录（或设置 `LLAMA_MODEL_PATH` 指向其它路径）。
   - 需要包含配置文件与权重文件（例如 `config.json`, `tokenizer.json`, `pytorch_model-00001-of-00002.bin` 等）。

3. **准备 gRPC 代码**
   - 在 `tee_gpu/` 目录下执行 `make msg_pb2.py msg_pb2_grpc.py`（或直接运行下文脚本，脚本会自动执行）。

## 基准脚本

### 1. 拆分推理（TEE + GPU）

```bash
cd tee_gpu/benchmarks
bash run_split_benchmark.sh
```

脚本执行流程：
1. 导出常用环境变量（模型路径、Prompt 文件、批大小等）；
2. 自动生成默认 Prompt 样例；
3. `make SGX=1 …` 构建 Gramine manifest 与签名；
4. 在宿主环境启动 `server.py`，监听 `localhost:50051`；
5. 使用 `gramine-sgx` 在 TEE 中运行 `tee_runner.manifest.sgx`，完成拆分推理并生成 `tee_gpu_benchmark.json`。

### 2. 纯 TEE 推理

```bash
cd tee_only_llama/benchmarks
bash run_full_tee_benchmark.sh
```

脚本执行流程：
1. 构建 `tee_runner` 的 Gramine manifest；
2. 在 TEE 内运行纯 CPU（无 GPU 卸载）的 LLaMA 推理；
3. 输出保存于 `tee_only_results.json`。

### 3. 拆分 vs 纯 TEE 对比

```bash
cd tee_only_llama/benchmarks
bash compare_split_vs_tee.sh
```

该脚本依次调用上述两个基准，最终使用 `jq` 汇总结果，输出包含两种模式耗时与生成结果的 JSON。

## 重要环境变量

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `LLAMA_MODEL_PATH` | `./weights/llama3.2-1b` | 模型权重目录 |
| `LLAMA_GPU_ENDPOINT` | `localhost:50051` | GPU 服务端监听地址 |
| `LLAMA_PROMPT_PATH` | `tee_gpu/prompts.txt` 或 `tee_only_llama/prompts.txt` | 输入 Prompt 文本路径 |
| `LLAMA_TEE_BATCH_SIZE` | `4` | 基准运行的批大小 |
| `LLAMA_MAX_LENGTH` | `256` | 文本生成最大长度 |
| `LLAMA_TEMPERATURE` | `0.7` | 采样温度 |
| `LLAMA_TOP_P` | `0.9` | nucleus sampling 截断概率 |
| `LLAMA_GPU_RESULT_PATH` | `tee_gpu/tee_gpu_benchmark.json` | 拆分模式结果输出 |
| `LLAMA_TEE_RESULT_PATH` | `tee_only_llama/tee_only_results.json` | 纯 TEE 模式结果输出 |

可通过导出环境变量覆盖默认值，以适应不同实验需求。

## Gramine 构建说明

- `tee_gpu/Makefile` 封装了 gRPC 代码生成与 manifest 构建逻辑。
- 执行 `make SGX=1 server.manifest.sgx server.sig tee_runner.manifest.sgx tee_runner.sig` 将生成：
  - `server.manifest.sgx` / `server.sig`：若希望在 Gramine 内运行 GPU 侧服务可使用；
  - `tee_runner.manifest.sgx` / `tee_runner.sig`：TEE 内推理程序。
- manifest 模板中已预配置必要挂载与白名单，如需新增文件访问，请修改模板后重新 `make`。

## 结果文件

- `tee_gpu/tee_gpu_benchmark.json`：拆分推理的耗时与生成结果。
- `tee_only_llama/tee_only_results.json`：纯 TEE 推理的耗时与生成结果。
- `compare_split_vs_tee.sh` 运行后，会在终端输出包含 `split` 与 `tee_only` 两部分的 JSON，用于对比性能差异。

## 常见问题

1. **gRPC 代码未生成**：确保在目标环境安装 `grpcio-tools`，并在 `tee_gpu/` 执行 `make msg_pb2.py`。
2. **Gramine 构建失败**：确认安装好 Gramine（及 SGX 驱动），并根据平台需求设置 `GRAMINE_LOG_LEVEL`、`ARCH_LIBDIR` 等环境。
3. **模型文件缺失**：检查 `LLAMA_MODEL_PATH` 是否指向包含 `config.json`、`tokenizer.json`、权重文件的目录。
4. **GPU 服务连接失败**：确保 `server.py` 已在宿主机运行，且 `LLAMA_GPU_ENDPOINT` 与监听地址一致。

## 许可证

项目继承原仓库许可，详见根目录 `LICENSE` 文件。
