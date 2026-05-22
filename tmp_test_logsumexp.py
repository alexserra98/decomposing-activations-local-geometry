"""
Verify that _distributed_logsumexp has the correct gradient in single-process mode
(it falls back to torch.logsumexp when dist is not initialized).
"""
import sys
sys.path.insert(0, "src")
import torch
from dalg.models.mfa import _distributed_logsumexp

torch.manual_seed(0)

# --- Test 1: dim=1 (log_num case: (B, K_local)) ---
B, K = 6, 10
x = torch.randn(B, K, requires_grad=True)

ref = torch.logsumexp(x, dim=1).sum()
ref.backward()
ref_grad = x.grad.clone()
x.grad = None

y = _distributed_logsumexp(x, dim=1).sum()
y.backward()

assert torch.allclose(x.grad, ref_grad, atol=1e-6), (
    f"dim=1 gradient mismatch\nmax_diff={( x.grad - ref_grad).abs().max():.2e}"
)
print("dim=1 gradient: OK")

# --- Test 2: dim=0 (log_den case: (K_local,)) ---
x2 = torch.randn(K, requires_grad=True)

ref2 = torch.logsumexp(x2, dim=0)
ref2.backward()
ref_grad2 = x2.grad.clone()
x2.grad = None

y2 = _distributed_logsumexp(x2, dim=0)
y2.backward()

assert torch.allclose(x2.grad, ref_grad2, atol=1e-6), (
    f"dim=0 gradient mismatch\nmax_diff={(x2.grad - ref_grad2).abs().max():.2e}"
)
print("dim=0 gradient: OK")

# --- Test 3: forward value matches ---
x3 = torch.randn(B, K)
ref3 = torch.logsumexp(x3, dim=1)
got3 = _distributed_logsumexp(x3, dim=1)
assert torch.allclose(ref3, got3, atol=1e-6), "forward value mismatch"
print("forward value:  OK")

print("All logsumexp checks passed.")
