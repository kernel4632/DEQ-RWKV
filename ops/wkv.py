import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_CUDA_EXT_LOADED = {}


def _load_cuda_ext(head_size, dtype):
    key = (head_size, dtype)
    if key in _CUDA_EXT_LOADED:
        return

    is_bf16 = dtype == torch.bfloat16
    chunk_len = 16

    flags = ["-res-usage", f"-D_N_={head_size}", f"-D_CHUNK_LEN_={chunk_len}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]

    extra_cflags = []
    if not is_bf16:
        flags.append("-D_FP32_")
        extra_cflags.append("-D_FP32_")

    # 加载 CUDA 扩展
    load(
        name=f"rwkv7_clampw_{'bf16' if is_bf16 else 'fp32'}_h{head_size}",
        sources=["cuda/rwkv7_clampw.cu", "cuda/rwkv7_clampw.cpp"],
        is_python_module=False,
        verbose=False,  # 设为 False 减少输出，如需调试可改为 True
        extra_cflags=extra_cflags,
        extra_cuda_cflags=flags,
    )
    _CUDA_EXT_LOADED[key] = True


class _RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, H, C = r.shape
        assert T % 16 == 0, "序列长度 T 必须能被 16 整除"
        assert all(i.is_contiguous() for i in [r, w, k, v, a, b]), "输入张量必须是连续的"

        y = torch.empty_like(v)
        s = torch.empty(B, H, T // 16, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)

        torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
        ctx.save_for_backward(r, w, k, v, a, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.is_contiguous(), "梯度 dy 必须是连续的"
        r, w, k, v, a, b, s, sa = ctx.saved_tensors
        dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
        torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
        return dr, dw, dk, dv, da, db


def wkv_cuda(r, w, k, v, a, b, head_size=64):
    B, T, HC = r.shape
    _load_cuda_ext(head_size, r.dtype)
    r, w, k, v, a, b = [x.view(B, T, HC // head_size, head_size).contiguous() for x in [r, w, k, v, a, b]]
    y = _RWKV7_CLAMPW_CUDA_OP.apply(r, w, k, v, a, b)
    return y.view(B, T, HC)


def wkv_cpu(r, w, k, v, a, b, head_size=64):
    B, T, C = r.shape
    H, N = C // head_size, head_size

    shape = (B, T, H, N)
    r, k, v, a, b = [x.view(shape) for x in [r, k, v, a, b]]
    w = -F.softplus(-w) - 0.5
    w = torch.exp(-torch.exp(w.view(shape)))

    state = torch.zeros(B, H, N, N)
    out = torch.empty_like(r)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C)


wkv = [wkv_cpu, wkv_cuda][torch.cuda.is_available()]


if __name__ == "__main__":
    print("开始测试 RWKV7 CPU 操作...")

    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)

    # 定义测试参数
    BATCH_SIZE = 2
    SEQ_LEN = 32
    NUM_HEADS = 4
    HEAD_SIZE = 64
    CHANNELS = NUM_HEADS * HEAD_SIZE
    DTYPE = torch.float32

    # 生成输入数据
    #    w 通常需要是负数，经过 -F.softplus(-w) - 0.5 后依然是负数
    #    再经过 exp 后变成 0~1 之间的衰减因子。如果 w 太大，exp(-exp(w)) 会变成 0，导致 state 不衰减。
    #    如果 w 太小，exp(-exp(w)) 会变成 1，导致 state 不更新。
    #    这里我们生成 [-1, 1] 范围的随机数，比较安全。
    
    r_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.1
    w_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.5 - 1.0 # 偏向负值
    k_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.1
    v_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.1
    a_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.1
    b_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.1

    # 启用梯度
    r_cpu.requires_grad_(True)
    w_cpu.requires_grad_(True)
    k_cpu.requires_grad_(True)
    v_cpu.requires_grad_(True)
    a_cpu.requires_grad_(True)
    b_cpu.requires_grad_(True)

    print(f"输入形状: {r_cpu.shape}")
    print(f"输入 w 范围: min={w_cpu.min():.4f}, max={w_cpu.max():.4f}")

    try:
        # --- 前向传播测试 ---
        print("\n执行前向传播...")
        y_cpu = wkv_cpu(r_cpu, w_cpu, k_cpu, v_cpu, a_cpu, b_cpu, head_size=HEAD_SIZE)
        
        # 检查输出是否有效
        if torch.isinf(y_cpu).any():
            raise ValueError("前向传播输出包含 inf")
        if torch.isnan(y_cpu).any():
            raise ValueError("前向传播输出包含 nan")
            
        print("前向传播成功。")
        print(f"输出形状: {y_cpu.shape}")
        print(f"输出范数: {y_cpu.norm():.4f}")
        print(f"输出范围: min={y_cpu.min():.4f}, max={y_cpu.max():.4f}")

        # --- 反向传播测试 ---
        print("\n执行反向传播...")
        # 生成一个随机的梯度，同样保持较小的数值范围
        dy_cpu = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS, dtype=DTYPE) * 0.1
        
        y_cpu.backward(dy_cpu)
        
        # 检查梯度是否有效
        grads = [r_cpu.grad, w_cpu.grad, k_cpu.grad, v_cpu.grad, a_cpu.grad, b_cpu.grad]
        grad_names = ['r', 'w', 'k', 'v', 'a', 'b']
        
        has_inf = False
        has_nan = False
        
        for name, grad in zip(grad_names, grads):
            if torch.isinf(grad).any():
                print(f"警告: {name} 的梯度包含 inf")
                has_inf = True
            if torch.isnan(grad).any():
                print(f"警告: {name} 的梯度包含 nan")
                has_nan = True
            print(f"{name} 梯度范数: {grad.norm():.4f}")

        if not has_inf and not has_nan:
            print("反向传播成功，所有梯度正常。")
        else:
            raise RuntimeError("反向传播检测到无效梯度")

    except Exception as e:
        print(f"\n测试失败，发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试流程结束")
