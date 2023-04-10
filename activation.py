import torch
from torch import nn

class ReLU_(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, constant):
        ctx.save_for_backward(input)
        ctx.constant = constant
        output = input.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        constant = ctx.constant
        grad_input = grad_output.clone()
        grad_input[input<0.] = 0.
        grad_input[input==0.] = constant * grad_input[input==0.]
        return grad_input, None

class LeakyReLU_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, constant, negative_slope=0.01):
        ctx.save_for_backward(input)
        ctx.constant = constant
        ctx.slope = negative_slope
        output = input.clamp(min=0)+input.clamp(max=0)*negative_slope
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        constant = ctx.constant
        slope = ctx.slope
        grad_input = grad_output.clone()
        grad_input = grad_input * (input > 0).float() + grad_input * (input < 0).float() * slope + grad_input * constant * (input==0).float()
        return grad_input, None, None

relu_ = ReLU_.apply
leakyrelu_ = LeakyReLU_.apply

class MyReLU(nn.Module):
    
    def __init__(self,constant):
        super().__init__()
        self.c = constant
        self.mask = torch.tensor(False)
        
    def forward(self, input):
        self.mask = (input==0).any()
        return relu_(input, self.c)

class MyLeakyReLU(nn.Module):
    
    def __init__(self,constant):
        super().__init__()
        self.c = constant
        self.mask = torch.tensor(False)
        
    def forward(self, input):
        self.mask = (input==0).any()
        return leakyrelu_(input, self.c)