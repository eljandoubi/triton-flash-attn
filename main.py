import torch
from src.ops.add import Add

if __name__ == "__main__":
    x = torch.randn((3, 64, 64, 5), device="cuda")
    y = torch.randn((3, 64, 64, 5), device="cuda")
    output = Add.apply(x, y)
    print(torch.abs(output - (x + y)).sum())
