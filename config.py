import torch
import torch.nn.functional as F
import math

test_every_x_epoch = 50
eps = 0.000001

# Aggregators
A_sum = lambda a: torch.sum(a)
A_cross_entropy = lambda a: torch.mean(torch.log(a + eps))
A_mean = lambda a: torch.mean(a) - 1
A_log_sigmoid = lambda a: F.logsigmoid(pa * (torch.sum(a) - a.size()[0] + 1 + pb0))
A_RMSE = lambda a: 1 - torch.sqrt(torch.mean((1 - a)**2.0) + eps) # Unclamped Yager/sum of squares
A_min = lambda a: torch.min(a)


# Normal product norm
T_P = lambda a, b: a * b
S_P = lambda a, b: a + b - a * b
I_RC = lambda a, b: S_P(1 - a, b)

# Godel norm
T_G = lambda a, b: torch.min(a, b)
S_G = lambda a, b: torch.max(a, b)
I_KD = lambda a, b: torch.max(1-a, b)

# # Sigmoid norm
# pa = 6
# pb0 = -0.5
# T_S = lambda a, b: F.sigmoid(pa * (a + b - 1 + pb0))
# S_S = lambda a, b: F.sigmoid(pa * (a + b + pb0))
# I_S = lambda a, c: F.sigmoid(pa * (1-a + c + pb0))

# Yager norm
tp = 2
ip = 2
T_Y = lambda a, b: torch.clamp(1 - ((1-a)**tp + (1-b)**tp + eps)**(1/tp), min=0)
S_Y = lambda a, b: torch.clamp((a**tp + b**tp)**(1/tp), max=1)

I_Y = lambda a, b: torch.clamp(((1-a)**ip + b**ip)**(1/ip), max=1)
I_RMSE = lambda a, b: (1/2 * ((1-a)**ip + b**ip))**(1/ip)
def _I_RY(a, c):
    r = 1 - ((1 - c)**ip - (1 - a)**ip + 0.1)**(1/ip)
    r[a <= c] = 1
    if (r != r).any():
        print('nan i')
    return r
I_RY = _I_RY

T_RMSE = lambda a, b: 1 - (1/2 *((1-a)**tp + (1-b)**tp) + eps)**(1/tp)

# Luk implications
I_L = lambda a, c: torch.clamp(1-a+c, max=1)

# Product norm implications
I_s_imp = lambda a, c: 1 - a + a * c
I_quad = lambda a, c: S_P(1 - a * a, 2 * c - c * c)

# Sigmoidal
pa = 9
pb0 = -0.5
# Product
T_SP = lambda a, b: torch.sigmoid(pa * (a * b + pb0))
# S_SP = lambda a, b: torch.sigmoid(pa * ((1 - a) * (1 - b) + pb0))

# General sigmoid
def sigmoid(f):
    def _sigmoid(a, b):
        # This is a bug: Doesnt work with f's that have a complete reach. If you use this for implications, it starts
        # acting up as for implications f(0, 0) and f(1, 1) are both 1
        if (a != a).any():
            print('nan a')
        if(b != b).any():
            print('nan b')
        f0 = 0#f(0, 0)
        f1 = 1#f(1, 1)
        y1 = math.exp(-pa * (f1 + pb0)) + 1
        y2 = math.exp(-pa * (f0 + pb0)) + 1
        r = (y1 / (y2 - y1)) * (y2 * torch.sigmoid(pa * (f(a, b) + pb0)) - 1)
        if (r != r).any():
            print('nan')
            import traceback
            traceback.print_stack()
        return r
    return _sigmoid

T_SP = sigmoid(lambda a, b: a * b)
S_SP = sigmoid(lambda a, b: 1 - (1-a) * (1 - b))
I_SP = sigmoid(lambda a, b: 1 - a + a * b)
I_SKD = sigmoid(lambda a, b: torch.max(1-a, b))
I_SY = sigmoid(I_Y)

def goguen(a, c):
    i = c / (a + eps)
    i[a <= c] = 1
    return i

def upper_contra_goguen(a, c):
    i1 = c/(a + eps)
    i2 = (1-a)/(1-c + eps)
    i = torch.max(i1, i2)
    i[a<=c] = 1
    return i
I_goguen = goguen
I_uGG = upper_contra_goguen

def normalized_rc(a, c):
    t_implication = 1 - a + a * c
    dMP = a / t_implication
    dMT = (1 - c) / t_implication
    tot_dMP = torch.sum(dMP)
    tot_dMT = torch.sum(dMT)
    return mu * c * dMP / tot_dMP + (1 - mu) * (1 - a) * dMT / tot_dMT

# Choice of norm
T = T_G
S = S_G

# Choice of implication
I = I_KD

# Choice of aggregator
A_quant = A_min
A_clause = lambda a, b, c: (a + b + c) / 3

rl_weight = 1.
same_weight = 1.
mu = 0.25

# If not mentioned, the value of s is 6
EXPERIMENT_NAME = 'split_100/full_godel'
EXPERIMENT_NAME += '_rlw_' + str(rl_weight) + '_samew_' + str(same_weight) + '_mu_' + str(mu)
print(EXPERIMENT_NAME)

lr = 0.01
momentum = 0.5
log_interval = 100
epochs = 80001