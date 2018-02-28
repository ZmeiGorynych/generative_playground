import torch

# multiplies all inputs by a fixed vector
class Net(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        if len(x.size())==2: # if got a single matrix
            return x @ self.w
        elif len(x.size())==3: # we got a batch of inputs
            return torch.matmul(x, self.w.view(1,-1,1))
    
    def input_shape(self):
        return (1,len(self.w))


