# 项目 8 · 容器化部署
# 构建：在【仓库根目录】执行（注意末尾的 . 是构建上下文）
#   docker build -t yunqi-llm-api -f stage06_deployment/project08_enterprise_api/Dockerfile .
# 运行（把 API Key 通过环境变量注入，不要写死在镜像里）：
#   docker run -p 8000:8000 -e SILICONFLOW_API_KEY=sk-xxx yunqi-llm-api

FROM python:3.12-slim

WORKDIR /app

# 先只拷依赖清单并安装——利用 Docker 层缓存：代码改动时不必重装依赖
COPY stage06_deployment/project08_enterprise_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 保留仓库目录结构拷贝，这样 main.py 里 parents[2] 仍能定位到 /app 下的 common 包
COPY common/ ./common/
COPY stage06_deployment/project08_enterprise_api/ ./stage06_deployment/project08_enterprise_api/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "stage06_deployment/project08_enterprise_api/main.py"]
