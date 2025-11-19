import mx

class LayerNorm(mx.Module):
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.gamma = mx.Tensor([emb_dim], init='ones')
        self.beta = mx.Tensor([emb_dim], init='zeros')

    def forward(self, x):
        # x: [batch, seq_len, emb_dim] or any shape ending in emb_dim
        return x.layer_norm(self.gamma, self.beta, self.eps)

class RMSNorm(mx.Module):
    def __init__(self, emb_dim, eps=1e-5):
        self.emb_dim = emb_dim
        self.eps = eps
        self.weight = mx.Tensor([emb_dim], init='ones')

    def forward(self, x):
        return x.rms_norm(self.weight, self.eps)