import torch

# 저장된 파일 로드
#activation = torch.load("llama3.2-1B.pt")
activation, *_ = torch.load("llama3.2-1B.pt")
print("📐 activation shape:", activation.shape)  # torch.Size([4499, 15, 8192])
print("✅ 파일 로드 성공")
print("📐 activation shape:", activation.shape)
print("🧠 activation dtype:", activation.dtype)
print("📊 activation 값 통계:")
print("   mean:", activation.mean().item())
print("   std :", activation.std().item())
print("   min :", activation.min().item())
print("   max :", activation.max().item())
