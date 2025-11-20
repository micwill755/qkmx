from .tensor import Tensor, zeros
from .functional import randn

class Linear:
    def __init__(self, d_in, d_out):
        self.weights = randn((d_in, d_out))
        self.bias = zeros((d_out,)) 
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # y = mx + b
        batch, seq_len = x.shape[0], x.shape[1]
        result = Tensor((batch, seq_len, self.weights.shape[-1]))
        for b in range(batch):
            result[b] = x[b].matmul(self.weights) + self.bias
        return result
