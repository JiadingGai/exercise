import torch
bs = 20
m = 128
n = 256
k = 8192

# Generate random numbers with the range [lo, hi]
lo, hi = -0.02, +0.02

X = (lo - hi) * torch.randn(bs, m, k) + hi
W = (lo - hi) * torch.randn(bs, k, n) + hi
y = torch.bmm(X, W)

# int8 quantization - per-channel (per outter channel)
nbits = 8
denom = 2 ** (nbits - 1) - 1

delta_X = torch.max(torch.abs(X), dim=2).values.unsqueeze(dim=2) / denom
delta_W = torch.max(torch.abs(W), dim=1).values.unsqueeze(dim=1) / denom

Xq = torch.round(X / delta_X).to(torch.int8)
Wq = torch.round(W / delta_W).to(torch.int8)

# need to convert Xq and Wq to int32 in torch.bmm to avoid integer overflow
yq = delta_X * torch.bmm(Xq.to(torch.int32), Wq.to(torch.int32)) * delta_W

rtol = 1e-07
atol = 1e-02
if not torch.allclose(y, yq, rtol=rtol, atol=atol):
    print("=========== y vs yq ===========")
    print(f"max diff: {(yq - y).abs().max().item()}")
    print(f"mean diff: {(yq - y).abs().mean().item()}")
    diff = (yq - y).abs()
    indices = torch.argwhere(diff >= diff.max())
    print(" y@max_diff = ", y[indices[0][0], indices[0][1], indices[0][2]])
    print("yq@max_diff = ", yq[indices[0][0], indices[0][1], indices[0][2]])

assert torch.allclose(y, yq, rtol=rtol, atol=atol)
print("PASS")
