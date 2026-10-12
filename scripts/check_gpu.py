import sys
import torch

print("Python 路径：", sys.executable)
print("PyTorch 版本：", torch.__version__)
print("PyTorch 构建使用的 CUDA：", torch.version.cuda)
print("CUDA 可用：", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("检查失败：PyTorch 无法使用 GPU，请保留以上输出。")

print("GPU：", torch.cuda.get_device_name(0))
print("计算能力：", torch.cuda.get_device_capability(0))
print("BF16 支持：", torch.cuda.is_bf16_supported())

free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"当前可用显存：{free_bytes / 1024**3:.2f} GiB")
print(f"总显存：{total_bytes / 1024**3:.2f} GiB")

# 不只检查设备是否可见，还实际执行一次 GPU 矩阵运算。
dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

with torch.inference_mode():
    x = torch.ones((512, 512), device="cuda", dtype=dtype)
    y = x @ x
    torch.cuda.synchronize()

    if not torch.all(y == 512).item():
        raise RuntimeError("GPU 矩阵运算结果不符合预期")

print("计算精度：", dtype)
print("结果所在设备：", y.device)
print("检查通过：GPU 矩阵运算正常。")