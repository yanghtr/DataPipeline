# ModelArts DP64 Direct Rewrite

## 背景

当前云道训练作业的 vLLM 启动脚本在 16 个 worker 上启动了 64 个独立 vLLM 实例：

```text
16 workers * 8 NPU / TP_SIZE 2 = 64 vLLM backends
```

每个 worker 本地有 4 个端口：

```text
16660, 16661, 16662, 16663
```

worker-0 额外启动了 nginx，监听：

```text
http://127.0.0.1:16669
```

nginx 的 `/etc/nginx/conf.d/vllm.conf` 里已经记录了所有 64 个真实 backend，例如：

```nginx
server 172.16.11.251:16660;
server 172.16.11.251:16661;
...
```

这个短期方案不 kill 已有服务，也不要求逐个打开 worker cloud shell。只在 worker-0 cloud shell 里运行 rewrite，代码会读取 nginx conf，提取 64 个 backend，然后直接请求这些 vLLM 实例，绕过 nginx 的 `16669` 转发路径。

## 新配置

配置文件：

```text
html_rewrite/configs/default_modelarts_dp64_nginx_conf.yaml
```

关键字段：

```yaml
backend_urls_from_nginx_conf: "/etc/nginx/conf.d/vllm.conf"
backend_url_path: "/v1/chat/completions"
num_workers_per_backend: 4
num_workers: 32
```

总并发计算方式：

```text
总并发 = backend 数量 * num_workers_per_backend
```

在当前 64 backend 作业里：

```text
num_workers_per_backend: 4  -> 总并发 256
num_workers_per_backend: 8  -> 总并发 512
num_workers_per_backend: 16 -> 总并发 1024
```

旧配置仍然可用。没有设置 `backend_urls_from_nginx_conf` 时，Stage 2 仍然只调用原来的单个 `url`。

注意：`num_workers` 仍然会影响 Stage 1 预处理；启用 DP direct 模式后，Stage 2 的 API 总并发由 `num_workers_per_backend` 控制。

## 使用方法

进入 worker-0 cloud shell，先确认 nginx conf 存在：

```bash
cat /etc/nginx/conf.d/vllm.conf
```

确认 worker-0 能直连某个远端 backend：

```bash
curl http://172.16.6.111:16660/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key-here" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

小样本测试：

```bash
cd /home/yanghaitao/Projects/DataPipeline
python -m html_rewrite.main \
  --config html_rewrite/configs/default_modelarts_dp64_nginx_conf.yaml \
  --stage all \
  --limit 100
```

全量运行：

```bash
cd /home/yanghaitao/Projects/DataPipeline
python -m html_rewrite.main \
  --config html_rewrite/configs/default_modelarts_dp64_nginx_conf.yaml \
  --stage all
```

如果已经有可复用的 Stage 1 预处理结果，把配置里的 `run_root_dir` 改成已有 run 目录，然后运行：

```bash
python -m html_rewrite.main \
  --config html_rewrite/configs/default_modelarts_dp64_nginx_conf.yaml \
  --stage rewrite
```

## 调参建议

先用：

```yaml
num_workers_per_backend: 4
```

确认无大量 `ConnectionError`、输出正常、64 个 backend 都有请求后，再尝试：

```yaml
num_workers_per_backend: 8
```

如果 8 仍然稳定，可以继续测 12 或 16。建议每次用 `--limit 100` 或 `--limit 1000` 先测一小段。

## 注意事项

这个方案只绕过 nginx 的请求转发，仍然复用 nginx conf 做 backend discovery。已有 vLLM 服务和 nginx 服务都不需要重启。

必须在 worker-0 cloud shell 运行，因为 `/etc/nginx/conf.d/vllm.conf` 只在 worker-0 上生成。

如果 worker-0 能访问 `127.0.0.1:16669`，但不能访问某些 `172.x.x.x:16660-16663` backend，需要先检查对应 worker 的 vLLM 进程和端口监听。
