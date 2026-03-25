FROM nvcr.io/nvidia/pytorch:26.03-py3

# 环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 1. 安装系统依赖 + 生成 locales
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    language-pack-zh-hans \
    fonts-noto-cjk \
    git curl wget vim nano zsh tmux htop \
    libsm6 libxext6 \
    && locale-gen zh_CN.UTF-8 \
    && update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# 2. 配置 pip 国内加速（阿里云）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set install.trusted-host mirrors.aliyun.com

# 3. 安装 Python 包（处理 NGC constraint + git 安装 Muon）
RUN : > /etc/pip/constraint.txt && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        lightning \
        matplotlib \
        polars \
        pyarrow \
        tensorboard \
        torchdeq \
        transformers && \
    pip install --no-cache-dir git+https://github.com/KellerJordan/Muon
    
# 4. 配置 Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    echo "export LANG=zh_CN.UTF-8" >> /root/.zshrc && \
    echo "export LC_ALL=zh_CN.UTF-8" >> /root/.zshrc && \
    echo "export TZ=Asia/Shanghai" >> /root/.zshrc

# 设置工作目录
WORKDIR /workspace

# 默认使用 zsh
SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]