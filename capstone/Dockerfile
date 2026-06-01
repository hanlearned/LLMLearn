# Capstone 容器化
# 构建（在【仓库根目录】执行，注意末尾的 . ）：
#   docker build -t yunqi-capstone -f capstone/Dockerfile .
# 运行：
#   docker run -p 8000:8000 -e SILICONFLOW_API_KEY=sk-xxx yunqi-capstone
#   或用 docker compose（见 capstone/docker-compose.yml）

FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用层缓存）
COPY capstone/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 保留目录结构拷贝：common 在 /app/common，capstone 在 /app/capstone
COPY common/ ./common/
COPY capstone/ ./capstone/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "capstone/backend/api.py"]
