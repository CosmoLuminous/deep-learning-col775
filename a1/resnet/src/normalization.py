import torch
import torch.nn as nn


class NoNorm(nn.Module):
    def __init__(self):
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 0
        self.eps = torch.tensor(eps)
        self.num_features = num_features
        self.affine = affine
        shape = (1, self.num_features, 1, 1)

        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        self._param_init()


    def _param_init(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, x):

        if self.training:            
                
            n = x.numel() / x.size(1)
            var = x.var(dim=(0,2,3), keepdim=True, unbiased=False)
            mean = x.mean(dim=(0,2,3), keepdim=True)

            with torch.no_grad():
                
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * (n/(n-1)) * var + (1 - self.momentum) * self.running_var

        else:
            # print(self.running_mean, self.running_var)
            mean = self.running_mean
            var = self.running_var

        x = (x - mean)/ torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.gamma + self.beta

        return x


class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceNorm, self).__init__()
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 0
        self.eps = torch.tensor(eps)
        self.num_features = num_features
        self.affine = affine
        shape = (1, self.num_features, 1, 1)

        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        self._param_init()


    def _param_init(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)


    def forward(self, x):
        N, C, H, W = x.shape

        assert C == self.num_features
        # print(x.shape)
        # x = x.view(N, self.num_features, -1)
        # print(x.shape, "\n")
        if self.training:
            mean = x.mean(dim=(2,3), keepdim=True)
            var = x.var(dim=(2,3), keepdim=True)
            # print(mean.shape, var.shape)
            # with torch.no_grad():
                
            #     self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            #     self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var

        else:
            # mean = self.running_mean
            # var = self.running_var            
            mean = x.mean(dim=(2,3), keepdim=True)
            var = x.var(dim=(2,3), keepdim=True)

        x = (x - mean)/ torch.sqrt(var + self.eps)

        # x = x.view(N, C, H, W)

        if self.affine:
            x = x * self.gamma + self.beta

        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(LayerNorm, self).__init__()
        self.eps = torch.tensor(eps)
        self.num_features = num_features
        self.affine = affine
        shape = (1, self.num_features, 1, 1)

        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        self._param_init()


    def _param_init(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, x):
        
        N, C, H, W = x.shape

        assert C == self.num_features

        if self.training:
            mean = x.mean(dim=(1,2,3), keepdim=True)            
            var = x.var(dim=(1,2,3), keepdim=True)
        else:
            mean = x.mean(dim=(1,2,3), keepdim=True)            
            var = x.var(dim=(1,2,3), keepdim=True)

        x = (x - mean)/ torch.sqrt(var + self.eps)

        if self.affine:
            x = x * self.gamma + self.beta

        return x



class GroupNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, group=8, affine=True):
        super(GroupNorm,self).__init__()
        self.eps = torch.tensor(eps)
        self.num_features = num_features
        self.affine = affine
        self.group = group        
        shape = (1, self.num_features, 1, 1)

        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        self._param_init()


    def _param_init(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, x):
        N, C, H, W = x.shape

        assert C % self.group == 0
        assert self.num_features == C

        x = x.view(N, self.group, C // self.group, H, W)
        
        mean = x.mean(dim=(1,2,3), keepdim=True)
        var = x.var(dim=(1,2,3), keepdim=True)

        x = (x - mean)/ torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        
        if self.affine:
            x = x * self.gamma + self.beta

        return x

class BatchInstanceNorm(nn.Module):
    def __init__(self, num_features, momentum = 0.1, eps=1e-5, rho=0.5, affine=True):
        super(BatchInstanceNorm, self).__init__()
        self.momentum = momentum
        self.running_mean = 0
        self.running_var = 0
        self.eps = torch.tensor(eps)
        self.num_features = num_features
        self.affine = affine
        self.rho = rho
        shape = (1, self.num_features, 1, 1)

        if self.affine:
            self.gamma = nn.Parameter(torch.empty(shape))
            self.beta = nn.Parameter(torch.empty(shape))

        self._param_init()


    def _param_init(self):
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    
    def forward(self, x):
        if self.training:            
                
            n = x.numel() / x.size(1)
            var_bn = x.var(dim=(0,2,3), keepdim=True, unbiased=False)
            mean_bn = x.mean(dim=(0,2,3), keepdim=True)

            with torch.no_grad():
                
                self.running_mean = self.momentum * mean_bn + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * (n/(n-1)) * var_bn + (1 - self.momentum) * self.running_var

        else:
            mean_bn = self.running_mean
            var_bn = self.running_var

        x_bn = (x - mean_bn)/ torch.sqrt(var_bn + self.eps)

        mean_in = x.mean(dim=(2,3), keepdim=True)
        var_in = x.var(dim=(2,3), keepdim=True)

        x_in = (x - mean_in)/ torch.sqrt(var_in + self.eps)

        x = self.rho * x_bn + (1-self.rho) * x_in

        if self.affine:
            x = x * self.gamma + self.beta

        return x