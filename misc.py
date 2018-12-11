import torch

def print_hook_mean(writer, name, step):
    def _hook(grad):
        writer.add_scalar(name, torch.mean(grad), step)
    return _hook
def print_hook_std(writer, name, step):
    def _hook(grad):
        writer.add_scalar(name, torch.std(grad), step)
    return _hook
def add_print_hooks(tensor, writer, name, step):
    tensor.register_hook(print_hook_mean(writer, name + '/mean', step))
    tensor.register_hook(print_hook_std(writer, name + '/stdev', step))

def one_hot(logits, labels):
    one_hot1 = logits.new_zeros(logits.size())
    one_hot1 = one_hot1.scatter(1, labels.unsqueeze(1), 1)
    return one_hot1.byte()

def cross_prod(x):
    n = x.size()[0]
    pairs = []
    for i in range(n):
        repeat_i = x[i].repeat(n)
        pairs.append(torch.stack([x, repeat_i], dim=1))
    return torch.cat(pairs, dim=0)