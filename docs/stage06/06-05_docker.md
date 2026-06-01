# 06-05 Docker 容器化：把 LLM 服务打包成「到处能跑」的镜像

> 🎯 **一句话**：用 Dockerfile 把你的 FastAPI LLM 服务连同 Python 依赖一起打包成镜像，做到「本地能跑 = 服务器能跑」，并通过环境变量在运行时安全注入 API Key——这是部署上线的标准交付物。

---

## 为什么需要它

「我机器上明明跑得好好的」是部署的经典噩梦：服务器 Python 版本不同、缺依赖、环境变量没配……

Docker 把**应用 + 依赖 + 运行环境**封进一个不可变镜像，任何装了 Docker 的机器 `docker run` 就能起，彻底消除环境差异。API Key 这类敏感配置则**不打进镜像**，而在运行时通过环境变量注入——既能跑，又不泄密。

---

## 核心用法

### 1. Dockerfile

```dockerfile
# 用 slim 版基础镜像：体积小、攻击面小
FROM python:3.11-slim

WORKDIR /app

# 先单独拷 requirements 再装依赖：利用 Docker 层缓存，
# 代码改动不会让依赖层失效，重建快
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷源码（变动频繁的放后面）
COPY . .

# 声明服务端口（仅文档作用，实际映射靠 -p）
EXPOSE 8000

# 启动命令：用 uvicorn 跑 FastAPI；host 必须 0.0.0.0 才能被容器外访问
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**逐块讲解：**
- **`python:3.11-slim`**：精简基础镜像，比完整版小几百 MB，构建快、漏洞面小。
- **先 COPY requirements 再 pip install**：Docker 按层缓存，依赖层只在 `requirements.txt` 变时才重建。只改业务代码时复用依赖层，构建秒级完成——这是核心优化。
- **`--no-cache-dir`**：不留 pip 缓存，进一步缩小镜像。
- **`EXPOSE 8000`**：声明容器监听端口，文档性质。
- **`--host 0.0.0.0`**：关键！默认 `127.0.0.1` 只容器内可达，必须 `0.0.0.0` 才能从宿主机/外部访问。

### 2. .dockerignore

```
venv/
__pycache__/
*.pyc
.env            # 绝不把含密钥的 .env 打进镜像！
.git/
embed_cache/
*.md
```

**本质在干什么？** `.dockerignore` 排除不该进镜像的文件——尤其 **`.env`（含 API Key）绝不能打进镜像**，否则镜像一旦泄露密钥即泄露。同时排除 venv/.git 等大目录，加快构建、缩小体积。

### 3. 构建与运行（运行时注入 API Key）

```bash
# 构建镜像
docker build -t llm-service:latest .

# 运行：-e 注入环境变量，-p 映射端口
docker run -d -p 8000:8000 \
  -e SILICONFLOW_API_KEY="sk-xxxxxx" \
  --name llm-svc llm-service:latest

# 或从本地 .env 文件批量注入（文件不进镜像，仅运行时读取）
docker run -d -p 8000:8000 --env-file .env --name llm-svc llm-service:latest
```

**逐块讲解：**
- **`-e KEY=value`**：运行时把 API Key 作为环境变量注入容器，`load_dotenv()` 之后 `os.getenv` 照样读得到。密钥与镜像分离，安全。
- **`--env-file .env`**：批量从本地 `.env` 注入，文件留在宿主机、不进镜像。
- **`-p 8000:8000`**：宿主机 8000 映射到容器 8000，外部才能访问。
- **`-d`**：后台运行。`docker logs llm-svc` 看日志，`docker stop/rm` 管理生命周期。

---

## 关键原理 / 实践要点

1. **依赖与代码分层**：先 COPY+install 依赖、再 COPY 代码，最大化利用层缓存，让改代码后的重建变快。这是写 Dockerfile 的第一优化。
2. **密钥永不进镜像**：`.env` 写进 `.dockerignore`，密钥一律运行时用 `-e`/`--env-file` 注入。生产更进一步用密钥管理服务（K8s Secret、Vault）。
3. **`--host 0.0.0.0` 必加**：忘了它会出现「容器在跑但外部连不上」的经典坑。
4. **slim 而非 alpine**：Python 在 alpine（musl libc）上常因编译型依赖出问题，`slim` 兼容性更好，是 Python 服务的稳妥选择。
5. **多阶段构建（进阶）**：需要编译依赖时用多阶段 build，构建产物拷进干净的运行镜像，进一步瘦身。
6. **配合 uvicorn 生产参数**：可加 `--workers` 跑多进程，或用 gunicorn 管理 uvicorn worker 提升并发。

---

## 你来改

- [ ] 故意把 `--host` 去掉用默认值，`docker run` 后从宿主机 curl，复现「连不上」，再加回 `0.0.0.0` 验证修复。
- [ ] 只改一行业务代码重新 `docker build`，观察依赖层是否命中缓存（构建是否秒级完成）。
- [ ] 用 `--env-file .env` 启动，进容器 `docker exec` 跑 `env | grep API`，确认密钥被注入但镜像里查不到 `.env`。

---

## 面试怎么考

**Q：Docker 化 LLM 服务，Dockerfile 关键点有哪些？**
A：用 python slim 基础镜像（小、兼容好）；先 COPY requirements 再 pip install、最后 COPY 代码，利用层缓存让改代码后重建快；EXPOSE 端口；CMD 用 uvicorn 启动且 `--host 0.0.0.0`（否则容器外访问不到）。配 .dockerignore 排除 venv/.git/.env 等。

**Q：API Key 怎么处理才安全？**
A：绝不打进镜像。把 .env 写进 .dockerignore，运行时用 `docker run -e KEY=value` 或 `--env-file .env` 注入环境变量，代码用 os.getenv 读取。密钥与镜像分离，镜像泄露不等于密钥泄露；生产进一步用 K8s Secret / Vault 等密钥管理。

**Q：为什么先拷 requirements 再拷代码？为什么 host 要 0.0.0.0？**
A：Docker 镜像按层缓存，依赖层只在 requirements 变化时重建。先装依赖、后拷代码，则仅改业务代码时复用依赖层、构建秒级完成。`--host 0.0.0.0` 让 uvicorn 监听所有网卡，容器外才能访问；默认 127.0.0.1 只容器内可达，会导致「服务在跑却连不上」。
