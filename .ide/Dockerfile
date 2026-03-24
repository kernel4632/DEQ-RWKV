# 使用 NVIDIA 官方镜像，CUDA 12.1, Ubuntu 22.04
# 自带 Python 3.10，完美兼容 PyTorch 官方 wheel 包
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装系统基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 语言和编码支持
    locales \
    language-pack-zh-hans \
    fonts-noto-cjk \
    # Python 构建工具 (不安装 python 本身，使用镜像自带的 3.10)
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    nano \
    # 【修改】在这里添加了 zsh
    zsh \
    tmux \
    htop \
    build-essential \
    cmake \
    ninja-build \
    # 深度学习常用系统库
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置中文环境变量
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 2. 安装 uv (极速 Python 包管理器)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 3. 配置 Python 环境
# 使用 uv 创建虚拟环境，并继承系统 python (3.10)
RUN uv venv /opt/venv

# 【关键修改 1】将 PATH 设置为 ENV，而不是仅在当前 RUN 有效
# 这样 ENTRYPOINT 脚本启动时也能找到 uv 和 python
ENV PATH="/opt/venv/bin:$PATH"

# 升级基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. 配置 pip 加速
# 设置阿里云镜像为默认源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set install.trusted-host mirrors.aliyun.com

# 5. 安装 PyTorch (CUDA 12.1)
# 强制从官方源下载，确保是 GPU 版本
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 恢复默认源为阿里云，用于后续安装其他库
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 6. 配置 Jupyter Lab (作为备用)
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_server_config.py

# 7. 配置 Shell 环境
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
RUN echo "export PATH=/opt/venv/bin:\$PATH" >> /root/.zshrc

# 设置工作目录
WORKDIR /workspace

# 暴露端口
EXPOSE 8888 6006

# 8. 配置启动脚本 (适配 CNB 平台)
# 这个脚本会先执行 uv sync，然后执行平台传入的命令
RUN echo '#!/bin/bash\n\
# 确保虚拟环境在 PATH 中\n\
export PATH=/opt/venv/bin:$PATH\n\
\n\
# 如果存在 pyproject.toml，执行 uv sync\n\
if [ -f pyproject.toml ]; then\n\
    echo "Running uv sync..."\n\
    uv sync\n\
fi\n\
\n\
# 执行传入的命令 (CNB 平台的默认命令)\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
