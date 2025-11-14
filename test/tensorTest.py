import qkmx
from qkmx import Tensor

if __name__ == "__main__":
    # Matrix multiply: (2, 3) @ (3, 4) = (2, 4)
    a = Tensor((2, 3))
    b = Tensor((3, 4))
    c = a.matmul(b)
    print(f"Result shape: {c.shape}")  # (2, 4)
    print(c)

    t = qkmx.zeros((3, 4))
    r = qkmx.randn((2, 3))

    print(r)
