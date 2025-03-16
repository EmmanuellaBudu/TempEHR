import torch.nn.functional as F

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    #tiled_x = x.expand(x_size, y_size, dim)
    #tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (x - y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self,  x_recon, x, z, mu,logvar,alpha, delta):
        recons_loss= F.mse_loss(x_recon, x)*delta
        z_prior= torch.randn_like(z)
        mmd_loss= compute_mmd(z,z_prior)*alpha
        loss = recons_loss+mmd_loss
        return loss, recons_loss, mmd_loss
    
class TimeLoss(nn.Module):
    def __init__(self):
        super(TimeLoss, self).__init__()

    def forward(self,recon_time,time,beta):
        time_loss = F.mse_loss(recon_time, time.unsqueeze(2))
        
        loss = beta*time_loss
        return loss
