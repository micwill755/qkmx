import mx

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.data = param.data - self.lr * param.grad
    
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep
    
    def update(self, model):
        self.t += 1
        
        for i, param in enumerate(model.parameters()):
            if param.grad is None:
                continue
            
            # Initialize moments if first time
            if i not in self.m:
                self.m[i] = mx.zeros_like(param.data)
                self.v[i] = mx.zeros_like(param.data)
            
            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)