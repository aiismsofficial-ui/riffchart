"""Sanity check: CUDA available and bf16 matmul works on the GPU."""
import time
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 20
print(f"bf16 8192^3 matmul: {dt*1e3:.2f} ms  ({2*8192**3/dt/1e12:.1f} TFLOPS)")
ref = (a[:64].float() @ b[:, :64].float())
print("max rel err vs fp32:", ((c[:64, :64].float() - ref).abs().max() / ref.abs().max()).item())
