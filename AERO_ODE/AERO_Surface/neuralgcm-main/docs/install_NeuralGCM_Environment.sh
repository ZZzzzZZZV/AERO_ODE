#!/bin/bash
##############################################################################
#                                                                            #
#        JAX 0.4.29 + PyTorch 2.4.0 完美共存方案                             #
#                                                                            #
#        兼容 cuDNN 9.1，满足 chex/optax 依赖                                #
#                                                                            #
##############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          正在安装 JAX 0.4.29 + PyTorch 2.4.0...                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}错误: 请先激活conda环境${NC}"
    exit 1
fi

echo -e "${GREEN}当前环境: $CONDA_DEFAULT_ENV${NC}\n"

# 步骤1: 清理旧版本
echo -e "${YELLOW}[1/5] 清理旧版本...${NC}"
pip uninstall -y jax jaxlib nvidia-cudnn-cu12 torch torchvision torchaudio 2>/dev/null || true
echo -e "${GREEN}✓ 清理完成${NC}\n"

# 步骤2: 安装 PyTorch 2.4.0
echo -e "${YELLOW}[2/5] 安装 PyTorch 2.4.0...${NC}"
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
echo -e "${GREEN}✓ PyTorch 安装完成${NC}\n"

# 步骤3: 安装 JAX 0.4.29 (支持 cuDNN 9.1)
echo -e "${YELLOW}[3/5] 安装 JAX 0.4.29 (cuda12.cudnn91)...${NC}"
pip install jax==0.4.29 jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo -e "${GREEN}✓ JAX 安装完成${NC}\n"

# 步骤4: 安装 cuDNN 9.1 (两者共用)
echo -e "${YELLOW}[4/5] 安装 cuDNN 9.1.0...${NC}"
pip install nvidia-cudnn-cu12==9.1.0.70
echo -e "${GREEN}✓ cuDNN 安装完成${NC}\n"

# 步骤5: 配置环境变量
echo -e "${YELLOW}[5/5] 配置环境变量...${NC}"

# 创建自动配置脚本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cat > $CONDA_PREFIX/etc/conda/activate.d/jax_pytorch_final.sh << 'ACTIVATEOF'
#!/bin/bash
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)
if [ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
fi
ACTIVATEOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/jax_pytorch_final.sh << 'DEACTIVATEOF'
#!/bin/bash
if [ -n "$_OLD_LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
    unset _OLD_LD_LIBRARY_PATH
fi
unset XLA_PYTHON_CLIENT_PREALLOCATE
unset XLA_PYTHON_CLIENT_MEM_FRACTION
DEACTIVATEOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/jax_pytorch_final.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/jax_pytorch_final.sh

# 写入 .bashrc
if ! grep -q "# NeuralGCM JAX 0.4.29 Fix" ~/.bashrc; then
    cat >> ~/.bashrc << 'BASHRCEOF'

# NeuralGCM JAX 0.4.29 Fix
if [ "$CONDA_DEFAULT_ENV" = "NeuralGCM" ]; then
    CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)
    if [ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ]; then
        export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
    fi
fi
BASHRCEOF
fi

echo -e "${GREEN}✓ 配置完成${NC}\n"

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    🎉 安装完全成功！🎉                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "请运行以下命令验证："
echo ""
echo "    source ~/.bashrc"
echo "    conda deactivate && conda activate NeuralGCM"
echo "    python -c \"import jax; print('JAX:', jax.devices())\""
echo "    python -c \"import torch; print('Torch:', torch.cuda.is_available())\""
echo ""

