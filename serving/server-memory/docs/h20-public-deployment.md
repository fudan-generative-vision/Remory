# H20 远端部署与本机统一推理池

交给 H20 上的 Codex 时，可以直接使用下面这段任务：

> 请读取此文件和 `handoff/h20/pins.json`，在这台 4×H20 主机上部署本包的
> Remory SGLang 服务，并提供 HTTPS 公网入口及 Bearer API key。
> 本机统一推理池将把你的服务作为远端后端；Remory 张量和快照保留在 H20 侧。
> 先检查已有的官方模型缓存、可用 GPU、驱动、共享内存和公网入口。
> 使用本包的固定 SGLang 版本、overlay 和 online step 250 compressor；
> 按下面脚本完成安装、路径迁移、启动和真实协议验收，再设为持续运行服务。
> 普通 OpenAI 接口和原生 Remory 接口都要保留。不要用 stock SGLang 或旧版 step2320/14925
> 替代，也不要把 memory handle 在副本之间负载均衡。
> HF 下载只走 hf-mirror.com，并通过 download-direct 直连、绕过所有代理；
> 镜像失败就报告，不通过 VPN 重试大文件。
> 完成后返回下文列出的地址、容量、身份及验收记录。API key 写入权限 600 的文件，
> 通过私密文件交付，不写进聊天、日志或版本库。若缺少域名/公网入口或所需 compressor
> 文件，先完成可独立完成的准备，再说明缺少的具体信息。

## 架构与文件

```text
r-codex → 本机统一队列 → 本机 online250 推理后端
                    ↘ HTTPS + API key → H20 网关 → SGLang TP4
                                                   ├─ GPU Remory registry
                                                   └─ 持久快照目录
```

本机统一队列负责请求调度；H20 网关负责鉴权和远端容量保护。
普通无状态请求可以在同一模型的本地和远端后端间分配。
新版 Remory 会话只选择声明 `remory_server_memory_v1` 的后端，创建后固定路由；
父会话及其子会话共用同一个后端。两端升级后都提供这项能力。旧版 14925
服务和会话必须先排空或单独保留，不能混入 online250 池。远端中断时，有状态会话
等待恢复并从远端快照恢复，不把已有 handle 改发到另一台机器。

代码包不含模型权重或密钥。需要另外准备：

- 完整官方 `Qwen/Qwen3.8-27B`，revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。这是已有运行环境的 pin，保持视觉模块。
- online step 250 压缩器：优先使用下面固定 revision 的公开 safetensors；也支持
  本机原始 serving export（`config.json`、`manifest.json`、`compressor.pt`）。
  包内 `compressor-metadata/` 提供与本机兼容的 serving 配置，不含权重。
- 能从本机访问的真实 HTTPS 域名/平台公网入口。单纯监听 `0.0.0.0` 不能证明公网可达。

**2026-11-05 用户已选择两端统一升级至公开最新版 online step 250。**
仓库为 `fudan-generative-ai/Qwen3.8-27B-REMORY-1.9B`，锁定 revision
`5ada60e61d5425012b92a7c2b2796fc0b0f29241`。这是在 Stage-II-14925 基础上继续训练的
新阶段计数；Qwen 目标模型保持冻结。见固定 revision 的
[release.json](https://huggingface.co/fudan-generative-ai/Qwen3.8-27B-REMORY-1.9B/blob/5ada60e61d5425012b92a7c2b2796fc0b0f29241/release.json)。
该版已验证无损导出和 CPU 前向一致性，没有重测论文基准，不能把旧成绩记为新权重成绩。

源 `.pt` 文件 SHA-256：
`cf1fe99c2f51f330b4b93055b0da4f930e1b01d1841d0b9b9825a8e4dc386732`。
公开发布采用 safetensors 和 `remory.inference.v1` 配置，
本包通过 `--public-checkpoint` 支持这种格式。
脚本完整校验公开文件 SHA-256，再生成与本机兼容的 serving 配置；hook 原样加载
全部 117 个张量（包括 `output_embedding_rms`），不改名、不重新序列化权重。
公开文件的 SHA-256 为
`59d36bcec81c44bb314b63afea4dae79d8cda53a8918a5364143d8ecb2a499e5`；
它和上面的原始 `.pt` 文件 SHA 不同，因为容器格式不同。代码仅允许这个经过验证的
固定映射，不能用 manifest 自报的源 SHA 将其他模型伪装为 online250。
原始 serving config 记录的 `source_global_step: 14925` 是基础训练来源；
实际 online250 权重身份由上述 checksum 和 runtime 的 `remory_release` 标识。
不能仅凭那个基础 step 或模型显示名判断是否已经升级。

已由 H20 操作者确认其当前运行 `mocoV3/Qwen3.8-27B-REMORY-1.9B` revision
`aafd984cbecff137273c33cbee7d4fc2395f688b`、step2320，源 SHA 为
`c2c6efde58350ddb9fb98c7bdaa0e8bb0576be46581751ecd70cc198e67a1764`。
它与升级前本机在运行的 14925 都需要更新。14925 本身也是公开权重，旧 revision 为
`f24de2ac3996da36678cd5d196a7dc0dbbc3aa34`，并非私有版本。本文件所有命令现已锁定
online250，不下载可变的 `main`，不绕过身份校验。

准备脚本校验其完整字节及 contract/tokenizer/config 身份，只修改派生 export 的
`target_model` 路径。目标模型仍需验证为上述 revision 的未修改官方权重；协议中的
metadata identity 不是对全部模型权重的远程密码学证明。

优先复用已验证、revision 一致的官方模型缓存。确需下载时，使用随包的
`handoff/h20/download-direct`，通过 `hf-mirror.com` 直连并绕过代理：

```bash
sh handoff/h20/download-direct /srv/remory-stack/server/bin/hf download \
  Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir /srv/models/Qwen3.8-27B

sh handoff/h20/download-direct /srv/remory-stack/server/bin/hf download \
  fudan-generative-ai/Qwen3.8-27B-REMORY-1.9B \
  --revision 5ada60e61d5425012b92a7c2b2796fc0b0f29241 \
  --include config.json manifest.json model.safetensors \
  --local-dir /srv/models/online250-public
```

## 安装与启动

以下 `/srv/...` 是示例部署路径，可以整体换成当前用户可写的持久目录。
命令在解包后的代码根目录执行。需要 Linux x86_64、Python 3.12 的 venv/development
headers、git、C/C++ 编译器及可用的 NVIDIA 驱动。

```bash
python3.12 tools/install_h20_engine.py --prefix /srv/remory-stack --dry-run
sh handoff/h20/download-direct python3.12 tools/install_h20_engine.py \
  --prefix /srv/remory-stack
```

安装器复用本仓库的固定依赖和 CUDA 选择逻辑，只安装 server 环境、固定 SGLang
源码及 overlay，不安装本机任务执行环境、不下载模型、不启动 GPU 服务。
基础 SGLang commit 为 `f08726fd56c7ff6d8bd258f1545f98148fa4ef58`。
如果已有兼容的 CUDA 编译器，可指定 `--cuda-home /path/to/cuda`。
不要使用旧的 `stack.sh install`，它仍面向 step2320 发布包。

取得公开 compressor 和目标模型后：

```bash
python3.12 tools/prepare_h20_runtime.py \
  --prefix /srv/remory-stack \
  --public-checkpoint /srv/models/online250-public \
  --target /srv/models/Qwen3.8-27B \
  --output /srv/remory-deployment

/srv/remory-stack/server/bin/python tools/serve_server_memory.py \
  --state /srv/remory-deployment/runtime.json \
  --runtime-dir /srv/remory-runtime --devices 0,1,2,3 --port 30300 \
  --context 262144 --token-pool 262144 --max-running-requests 2 \
  --memory-fraction 0.75
```

如已有本机原始格式 export，可改用 `--checkpoint /srv/models/online250-original`。
两种方式生成相同的 portable artifact identity。架构配置也随包迁移，H20 不依赖
本机原有的绝对文件路径。

升级现有 H20 服务时，先在新的持久代码目录解包，准备新的 deployment 和
runtime 目录。若现有 prefix 的 SGLang commit、依赖和 CUDA 已符合本包 pin，可复用
它的解释器和基础源码；`serve_server_memory.py` 会另外生成隔离的 overlay。
不要在旧 prefix 上重复调用安装器绕过其所有权检查。不兼容时安装到新的 prefix。
准备完成后排空旧 H20 服务，停止它占用的四张卡，再更新 supervisor 启动命令。
保留旧代码、配置和权重用于回退；新模型使用新的快照目录，不能恢复 step2320 的
以及 14925 的 memory handle 或快照。随后重新执行下面的真实协议验收和容量测试。

先确认四张卡确实属于这次分配且未被其他作业占用。以上是待实测的初始容量，
不是已测得的 H20 性能。可以先用较小 context 做启动诊断，最终记录真实 context、
并发和显存余量；服务上报的限制会约束本机客户端。TP4 对应一个 HTTP 后端。
如改成多个独立副本，每个副本使用独立端口、快照目录和 backend 名称，逐个验收。

容器内检查 `/dev/shm`。本地曾因 64 MiB 共享内存导致 TP2 在 NCCL 初始化阶段失败；
NVIDIA 的容器示例使用 `--shm-size=1g --ulimit memlock=-1`。
按机器实际环境处理，不要复制本地故障诊断时关闭 P2P/IB 的临时开关。
参见 [NCCL 官方说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html#shared-memory)。

`prepare_h20_runtime.py` 保留原始 export，派生目录通过符号链接使用同一份
`compressor.pt` 或 `model.safetensors`，因此原目录必须保留。
`/srv/remory-runtime/memory` 必须持久化。
重启保持相同模型、路径、TP 拓扑、backend 路由和快照目录。

## 公网网关

生成密钥时只写文件，不输出内容：

```bash
python3.12 - <<'PY'
import os, secrets
fd = os.open('/srv/remory-deployment/api.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as out:
    out.write(secrets.token_urlsafe(48) + '\n')
PY

PYTHONPATH="$PWD/src" /srv/remory-stack/server/bin/python \
  -m recursive_memory.stack.gateway \
  --config handoff/h20/gateway.json --host 127.0.0.1 --port 30380 \
  --api-key-file /srv/remory-deployment/api.key
```

通过现有平台 HTTPS 入口或 `handoff/h20/Caddyfile` 反向代理到 `127.0.0.1:30380`。
使用 Caddy 时把 `REMORY_DOMAIN` 设置为实际域名，验证 DNS、证书及外部连通性，并先
运行 `caddy validate --config handoff/h20/Caddyfile --adapter caddyfile`。
引擎 `30300` 和网关 `30380` 保留 loopback；对公网只发布 HTTPS 入口。
网关会转发允许的推理和 Remory 路由，管理接口不会公开。

当前只有 `platform-domain.example` 的 code-server 网页入口：
`/proxy/30380` 需要网页登录，且已发现 HTTPS 证书域名不匹配，因此不能填为模型
`base_url`。需平台提供独立、证书有效的 HTTPS API 路由，或经受控公网中继建立
H20 主动连出的加密隧道。不能把 API key 发到该 HTTP 网页代理，也不能关闭证书校验。

保留 SSE 和断开取消语义。Caddy 对 `text/event-stream` 自动即时 flush；
不要设置 `flush_interval -1`，因为该模式会改变客户端断开时的取消行为。
不要启用生成 POST 的重试。见 [Caddy 官方流式代理说明](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming)。

引擎、网关及反向代理按目标主机现有的 systemd/容器服务方式持续运行。
重启不要删除快照；凭据轮换后重启远端网关。本机上游 key 文件每次请求都会重读。

## 验收与回传

先在远端经 loopback 网关验收，再从本机经真实公网地址验收：

```bash
/srv/remory-stack/server/bin/python tools/check_remote_memory.py \
  --base-url http://127.0.0.1:30380/backends/h20-tp4 \
  --api-key-file /srv/remory-deployment/api.key \
  --checkpoint /srv/remory-deployment/serving \
  --output /srv/remory-deployment/protocol-smoke.json
```

脚本做真实 encode、recursive encode、旧句柄释放、快照 save/load、compose、renew、
带 memory 的 generation 和 cleanup，不替代模型质量评测。另验证无 key 返回 401、
正确 key 可调用 `/v1/models` 和流式 chat、取消请求后容量归还。公网 TLS 和 TP4 都要
在 H20 实测；本地 CPU 测试不能代替这些步骤。

交回以下内容，key 单独通过私密文件交付：

```json
{
  "native_base_url": "https://实际域名/backends/h20-tp4",
  "openai_base_url": "https://实际域名/backends/h20-tp4/v1",
  "model": "Qwen3.8-27B",
  "max_inflight": 2,
  "context_length": 262144,
  "tp_size": 4,
  "artifact_identity": "填 get_server_info 的实际对象",
  "server_epoch": "填实际值",
  "protocol_smoke": "实际验收记录路径"
}
```

## 加入本机统一池

当前推荐使用 **server-owned memory + pinned backend**，见
[混合推理池与性能更新](inference-pool.md)。普通生成可以按容量分流；一旦会话
拥有远端 memory，父会话、子会话、compact 和恢复均使用同一后端。
正常请求只携带 handle/snapshot 引用，不传输 soft-token 张量值。

```bash
python tools/add_remote_backend.py \
  --config gateway.example.json \
  --base-url https://h20.example.org/backends/h20-tp4 \
  --api-key-file /secure/h20.key \
  --local-checkpoint /models/remory-online250 \
  --max-inflight 2 --memory-pinned-only \
  --output gateway.with-h20.json
```

本地和远端应部署同一公开 checkpoint。旧 `.pt` 与公开 safetensors 的文件哈希不同，
只有文档中明确验证过的固定 pair 可启用 `--verified-artifact-pair`；不得将任意
checkpoint 声称为等价。网关由操作方在排空后重启，工具只创建新配置。

11 月 6 日的 `stateless_only + allow_tensor_recover` 是旧式 CPU 张量传输路径，
[当时的验证记录](tensor-recovery-verification.json) 保留供追溯。
它不代表“只发送引用”，新的引用式配置保持 `allow_tensor_recover: false`。

## 推理性能开关

runtime 的 `server` 支持 `cuda_graph`、`overlap_schedule`、
`piecewise_cuda_graph` 和 `chunked_prefill_size`。例如启用前面三个布尔项并设置
`chunked_prefill_size: 8192`。未显式设置时保留原有保守行为。
须检查启动日志及实际 `/get_server_info`：模型可能自动禁用 piecewise prefill
graphs，这不影响 decode CUDA Graph。不能只据配置文件声称所有 graph 路径已启用。

残差编码需要一次完整源序列；图片与 soft memory 组合也需要完整图像坐标，
因此这两种预填充保留单请求完整执行及必要的结果同步。普通文本和文本记忆恢复
支持分块预填充，decode 可以使用 CUDA Graph。部署前应覆盖跨分块记忆位置、
递归编码和带图并发请求，升级时排空请求并保留原会话及实验进度。
