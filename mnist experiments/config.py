import torch
import torch.nn.functional as F
import math
from argparse import ArgumentParser


class Config(object):
    def __init__(self):
        self.t = None
        self.i = None
        self.a_quant = None
        self.rl_weight = None
        self.same_weight = None
        self.epochs = None
        self.name = None
        self.p = None
        self.s = None
        self.b0 = None

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)
        self.epochs += 1

        self.experiment_name = 'split_100/-n {}: -t {} -i {} -a {}'.format(self.name, self.t, self.i, self.a_quant)
        self.experiment_name += '-rlw {} samew {}'.format(str(self.rl_weight), str(self.same_weight))
        self.experiment_name += ' -s {} b0 {} p {}'.format(str(self.s), str(self.b0), str(self.p))

        print(self.experiment_name)

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

    def setup_parser(self):
        parser = ArgumentParser(description='training code')

        parser.add_argument('-t', dest='t', type=str, default='Y')
        parser.add_argument('-i', dest='i', type=str, default='SP')
        parser.add_argument('-a', dest='a_quant', type=str, default='cross_entropy')
        parser.add_argument('-rlw', dest='rl_weight', type=float, default=10.0)
        parser.add_argument('-samew', dest='same_weight', type=float, default=1.0)
        parser.add_argument('-epochs', dest='epochs', type=int, default=50000)
        parser.add_argument('-n', dest='name', type=str, default='')
        parser.add_argument('-p', dest='p', type=float, default=2.0)
        parser.add_argument('-s', dest='s', type=float, default=9.0)
        parser.add_argument('-b0', dest='b0', type=float, default=-0.5)

        return parser

    def choose_operators(self):
        self.T = CONJUNCTS[self.t]
        self.I = IMPLICATIONS[self.i]
        self.A_quant = AGGREGATORS[self.a_quant]


conf = Config()

test_every_x_epoch = 1000
eps = 0.000001

p_yager = conf.p
ap = p_yager

AGGREGATORS = {}
CONJUNCTS = {}
DISJUNCTS = {}
IMPLICATIONS = {}

# Aggregators
AGGREGATORS['sum'] = lambda a: torch.sum(a)
AGGREGATORS['cross_entropy'] = lambda a: torch.mean(torch.log(a + eps))
AGGREGATORS['mean'] = lambda a: torch.mean(a) - 1
AGGREGATORS['log_sigmoid'] = lambda a: F.logsigmoid(pa * (torch.sum(a) - a.size()[0] + 1 + pb0))
AGGREGATORS['RMSE'] = lambda a: 1 - torch.sqrt(torch.mean((1 - a)**2.0) + eps) # Unclamped Yager/sum of squares
# Adding random noise to make sure it arbitrarily chooses an argument when there are multiple lowest inputs
AGGREGATORS['min'] = lambda a: torch.min(a + torch.randn_like(a) / 1000)
AGGREGATORS['LK'] = lambda a: torch.clamp(torch.sum(a) - a.size()[0] + 1, min=0)
AGGREGATORS['T'] = lambda a: 2/math.pi * torch.asin(torch.prod(torch.sin(a * math.pi / 2)))
AGGREGATORS['H'] = lambda a: 1 / (1 + torch.sum((1-a) / (a+eps)))
AGGREGATORS['Y'] = lambda a: 1 - (torch.sum((1 - a)**ap) + eps)**(1/ap)

def upper_contra(I):
    def _implication(a, c):
        return torch.max(I(a, c), I(1-c, 1-a))
    return _implication

def lower_contra(I):
    def _implication(a, c):
        return torch.min(I(a, c), I(1-c, 1-a))
    return _implication

# Normal product norm
CONJUNCTS['P'] = lambda a, b: a * b
DISJUNCTS['P'] = lambda a, b: a + b - a * b

IMPLICATIONS['RC'] = lambda a, b: DISJUNCTS['P'](1 - a, b)
IMPLICATIONS['quad'] = lambda a, c: DISJUNCTS['P'](1 - a * a, 2 * c - c * c)

# Godel norm
CONJUNCTS['G'] = lambda a, b: torch.min(a, b)
DISJUNCTS['G'] = lambda a, b: torch.max(a, b)
IMPLICATIONS['KD'] = lambda a, b: torch.max(1-a, b)
def godel_i(a, c):
    i = c
    i[a <= c] = 1
    return i

IMPLICATIONS['G'] = godel_i
IMPLICATIONS['uG'] = upper_contra(IMPLICATIONS['G'])
IMPLICATIONS['lG'] = lower_contra(IMPLICATIONS['G'])

# # Sigmoid norm
# pa = 6
# pb0 = -0.5
# T_S = lambda a, b: F.sigmoid(pa * (a + b - 1 + pb0))
# S_S = lambda a, b: F.sigmoid(pa * (a + b + pb0))
# I_S = lambda a, c: F.sigmoid(pa * (1-a + c + pb0))

# Yager norm
tp = p_yager
ip = p_yager
CONJUNCTS['Y'] = lambda a, b: torch.clamp(1 - ((1-a)**tp + (1-b)**tp + eps)**(1/tp), min=0)
DISJUNCTS['Y'] = lambda a, b: torch.clamp((a**tp + b**tp + eps)**(1/tp), max=1)

IMPLICATIONS['Y'] = lambda a, b: torch.clamp(((1-a)**ip + b**ip + eps)**(1/ip), max=1)
IMPLICATIONS['RMSE'] = lambda a, b: (1/2 * ((1-a)**ip + b**ip + eps))**(1/ip)
def _I_RY(a, c):
    r = 1 - ((1 - c)**ip - (1 - a)**ip + eps)**(1/ip)
    r[a <= c] = 1
    if (r != r).any():
        print('nan i')
    return r
IMPLICATIONS['RY'] = _I_RY

CONJUNCTS['RMSE'] = lambda a, b: 1 - (1/2 *((1-a)**tp + (1-b)**tp) + eps)**(1/tp)

# Luk implications
CONJUNCTS['LK'] = lambda a, b: torch.clamp(a + b - 1, min=0)
DISJUNCTS['LK'] = lambda a, b: torch.clamp(a + b, max=1)
IMPLICATIONS['LK'] = lambda a, c: torch.clamp(1-a+c, max=1)

# Trigonometric
CONJUNCTS['T'] = lambda a, b: 2 / math.pi * torch.asin(eps+torch.sin(a * math.pi/2)*torch.sin(b * math.pi/2))
DISJUNCTS['T'] = lambda a, b: 2 / math.pi * torch.acos(eps+torch.cos(a * math.pi/2)*torch.cos(b * math.pi/2))
IMPLICATIONS['T'] = lambda a, c: DISJUNCTS['T'](1-a, c)

# Hamacher
v = 0
CONJUNCTS['H'] = lambda a, b: a * b / (eps + v + (1 - v)*(a + b - a * b))
DISJUNCTS['H'] = lambda a, b: (1 - (1 - v) * a * b)/(eps + 1 - (1 - v) * a * b)
IMPLICATIONS['H'] = lambda a, c: DISJUNCTS['H'](1-a, c)
# Sigmoidal
pa = conf.s
pb0 = conf.b0

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

CONJUNCTS['SP'] = sigmoid(lambda a, b: a * b)
DISJUNCTS['SP'] = sigmoid(lambda a, b: 1 - (1-a) * (1 - b))
IMPLICATIONS['SP'] = sigmoid(lambda a, b: 1 - a + a * b)
IMPLICATIONS['SLK'] = sigmoid(IMPLICATIONS['LK'])
IMPLICATIONS['SucLK'] = sigmoid(lambda a, b: 1-a+b)
IMPLICATIONS['SKD'] = sigmoid(lambda a, b: torch.max(1-a, b))
IMPLICATIONS['SY'] = sigmoid(IMPLICATIONS['Y'])

def goguen(a, c):
    i = c / (a + eps)
    i[a <= c] = 1
    return i

IMPLICATIONS['GG'] = goguen
IMPLICATIONS['uGG'] = upper_contra(IMPLICATIONS['GG'])
IMPLICATIONS['lGG'] = lower_contra(IMPLICATIONS['GG'])

def normalized_rc(a, c):
    t_implication = 1 - a + a * c
    dMP = a / t_implication
    dMT = (1 - c) / t_implication
    tot_dMP = torch.sum(dMP)
    tot_dMT = torch.sum(dMT)
    return mu * c * dMP / tot_dMP + (1 - mu) * (1 - a) * dMT / tot_dMT

# Choice of aggregator
A_clause = lambda a, b, c: (a + b + c) / 3

conf.choose_operators()

mu = 0.25

# If not mentioned, the value of s is 6
lr = 0.01
momentum = 0.5
log_interval = 500

