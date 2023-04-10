import torch
import warnings
from torch.nn import Parameter, Linear, Conv2d

def int2bit(integer_tensor, num_bits, device):
    
    dtype = integer_tensor.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype).to(device)
    exponent_bits = exponent_bits.repeat(integer_tensor.shape + (1,))
    out = integer_tensor.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2

def unif_bit2float(T, expected_bits, device):

    if expected_bits == 8:
        num_e_bits = 4
        num_m_bits = 3
        bias = 7

    elif expected_bits == 16:
        num_e_bits = 5
        num_m_bits = 10
        bias = 15

    elif expected_bits == 32:
        num_e_bits = 8
        num_m_bits = 23
        bias = 127
    
    else:
        warnings.warn("wrong expected bits. choose among [8,16,32]")

    dtype = torch.float64

    # sign
    s = int2bit(torch.randint(2,T,dtype=dtype).to(device), 1, device)
    # exponent bits 
    e = int2bit(torch.randint(bias,T,dtype=dtype).to(device), num_e_bits, device)
    # mantissa bits
    m = int2bit(torch.randint(2**num_m_bits,T,dtype=dtype).to(device), num_m_bits, device)

    out = ((-1) ** s).squeeze(-1).type(dtype)
    exponents = -torch.arange(-(num_e_bits - 1.), 1.).to(device)
    exponents = exponents.repeat(e.shape[:-1] + (1,))
    e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
    out *= 2 ** e_decimal
    matissa = (torch.Tensor([2.]) ** (
        -torch.arange(1., num_m_bits + 1.))).repeat(m.shape[:-1] + (1,)).to(device)
    out *= 1. + torch.sum(m * matissa, dim=-1)
    
    return out

def unif_weight_copy(model1, model2, precision, device):
    
    bias = model1.bias
    
    d2 = dict(model2.named_modules())

    for name, layer in model1.named_modules():
        if isinstance(layer, (Linear, Conv2d)):

            unif_weight= unif_bit2float(layer.weight.shape,precision, device)
            layer.weight = Parameter(unif_weight)
            d2[name].weight = Parameter(unif_weight.clone())

            if bias: 
     
                unif_bias = unif_bit2float(layer.bias.shape,precision, device)
                layer.bias = Parameter(unif_bias)
                d2[name].bias = Parameter(unif_bias.clone())

    return model1, model2
            

def touch0(model, act_fn):
    
    init = torch.tensor(False)

    for _, layer in model.named_modules():
        if isinstance(layer, act_fn):
            init += layer.mask.cpu()

    return int(init)


def diff_grad_chk(model1, model2):

    init = torch.tensor(False)
    bias = model1.bias
    dict_m2 = dict(model2.named_modules())

    for name,layer in model1.named_modules():
        if isinstance(layer,(Linear,Conv2d)):
            init += torch.ne(layer.weight.grad, dict_m2[name].weight.grad).any().cpu()
            if bias: init += torch.ne(layer.bias.grad, dict_m2[name].bias.grad).any().cpu()

    return int(init)


if __name__ == "__main__":
    x = torch.nn.Linear(4,8)
    y = torch.nn.Conv2d(200,100,20,20)
    a = unif_bit2float(x.weight.shape,8)
    b = unif_bit2float(y.weight.shape,32)
    x.weight = torch.nn.Parameter(a)
    y.weight = torch.nn.Parameter(b)
    print(x.weight,y.weight,sep='\n'+'*'*80+'\n')