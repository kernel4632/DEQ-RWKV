FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 安装 git、openssh-server、nvidia-cuda-toolkit、build-essential
RUN apt-get update && apt-get install -y curl gnupg2 && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" | tee /etc/apt/sources.list.d/nvidia-cuda.list && \
    apt-get update && \
    apt-get install -y \
        git \
        openssh-server \
        cuda-toolkit-12-4 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 启动时自动执行 uv sync
RUN echo '#!/bin/bash\ncd /workspace && uv sync\nexec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]