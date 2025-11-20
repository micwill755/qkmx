import mx

class LayerNorm(mx.Module):
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.gamma = mx.ones([emb_dim])
        self.beta = mx.zeros([emb_dim])

    def forward(self, x):
        # x: [batch, seq_len, emb_dim] or any shape ending in emb_dim
        return x.layer_norm(self.gamma, self.beta, self.eps)

class RMSNorm(mx.Module):
    def __init__(self, emb_dim, eps=1e-5):
        self.emb_dim = emb_dim
        self.eps = eps
        self.weight = mx.ones([emb_dim])

    def forward(self, x):
        return x.rms_norm(self.weight, self.eps)