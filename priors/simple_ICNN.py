import torch
import torch.nn as nn
from deepinv.optim import Prior
import torch.nn.utils.parametrize as P

class ZeroMean(nn.Module):
    """Enforcing zero mean on the filters improves performances"""

    def forward(self, x):
        return x - torch.mean(x, dim=(1, 2, 3), keepdim=True)

class ICNN_2l(nn.Module):
    def __init__(self,in_c,channels,kernel_size=5,smoothing=0.01):
        super(ICNN_2l, self).__init__()
        self.in_c = in_c
        self.channels = channels
        self.smoothing = smoothing
        self.padding = kernel_size // 2
        self.wx = nn.Conv2d(in_c,channels,kernel_size=kernel_size,padding=self.padding,bias=True)
        self.wz = nn.Conv2d(channels,channels,kernel_size=kernel_size,padding=self.padding,bias=True)
        self.scaling = nn.Parameter(torch.log(torch.tensor(0.001))*torch.ones(1,channels,1,1))
        self.act = lambda x: torch.clip(x, 0.0, self.smoothing)**2/(2*self.smoothing) + torch.clip(x, self.smoothing) - self.smoothing
        
        P.register_parametrization(self.wx, "weight", ZeroMean())
        
    def forward(self, x):
        self.zero_clip_weights()
        z1 = self.act(self.wx(x))
        z = self.act(self.wz(z1))*torch.exp(self.scaling)
        return (torch.sum(z.reshape(z.shape[0],-1),dim=1)).view(x.shape[0])
    
    def zero_clip_weights(self): 
        self.wz.weight.data.clamp_(0)
        return self 
    
    #Weight init
    def init_weight(self):
        wx = nn.init.xavier_normal_(self.wx.weight)
        self.wx.weight.data = wx.data-torch.mean(wx.data,(2,3),True)
        #self.wz.weight.data = torch.exp(-2*torch.rand_like(self.wz.weight.data))
        self.wz.weight.data.normal_(-10.0,0.1).exp_()
        return self   
    
class simple_ICNNPrior(Prior):
    def __init__(self,in_channels,channels,device,kernel_size=5,smoothing=0.01):
        super().__init__()
        self.icnn = ICNN_2l(in_channels,channels,kernel_size,smoothing).to(device)
        self.icnn.init_weight()
        
    def g(self, x):
        return self.icnn(x)
    
    def grad(self, x):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            z = torch.sum(self.g(x_))
            grad = torch.autograd.grad(z,x_,create_graph=True)[0]
        return grad