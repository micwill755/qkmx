import mx
import math

class LayerNorm(mx.Module):
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.scale = [1 for _ in range(emb_dim)]
        self.shift = [0 for _ in range(emb_dim)]

    def forward(self, x):
        # x is flattened: [token0_emb, token1_emb, ...]
        seq_len = len(x) // self.emb_dim
        result = [0 for _ in range(len(x))]
        
        for token_idx in range(seq_len):
            start_idx = token_idx * self.emb_dim
            end_idx = start_idx + self.emb_dim
            
            # Calculate mean for this token's embedding
            mean = sum(x[start_idx:end_idx]) / self.emb_dim
            
            # Calculate variance
            var = sum((x[i] - mean) ** 2 for i in range(start_idx, end_idx)) / self.emb_dim
            
            # Normalize and apply scale/shift
            for i in range(self.emb_dim):
                norm_val = (x[start_idx + i] - mean) / math.sqrt(var + self.eps)
                result[start_idx + i] = self.scale[i] * norm_val + self.shift[i]
        
        return result
    
class RMSNorm(mx.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weights = [1.0] * emb_dim

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        result = mx.Tensor((batch, seq_len, emb_dim))

        for b in range(batch):
            for t in range(seq_len):
                ms = sum(x[b][t][e] ** 2 for e in range(emb_dim)) / emb_dim
                rms = math.sqrt(ms + self.eps)
                for e in range(emb_dim):
                    result[b][t][e] = (x[b][t][e] / rms) * self.weights[e]
        
        return result