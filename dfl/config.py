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
        self.tp = None
        self.ip = None
        self.s = None
        self.b0 = None
        self.dds = None
        self.dsd = None
        self.ss = None
        self.g = None
        self.experiment_name = None

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)
        self.epochs += 1

        self.reset_experiment()

    def setup_parser(self):
        parser = ArgumentParser(description="training code")

        parser.add_argument("-t", dest="t", type=str, default="Y")
        parser.add_argument("-i", dest="i", type=str, default="SP")
        parser.add_argument("-a", dest="a_quant", type=str, default="cross_entropy")
        parser.add_argument("-rlw", dest="rl_weight", type=float, default=10.0)
        parser.add_argument("-samew", dest="same_weight", type=float, default=1.0)
        parser.add_argument("-epochs", dest="epochs", type=int, default=50000)
        parser.add_argument("-n", dest="name", type=str, default="")
        parser.add_argument("-tp", dest="tp", type=float, default=2.0)
        parser.add_argument("-ip", dest="ip", type=float, default=0.5)
        parser.add_argument("-s", dest="s", type=float, default=9.0)
        parser.add_argument("-b0", dest="b0", type=float, default=-0.5)
        parser.add_argument("-dds", dest="dds", type=float, default=1.0)
        parser.add_argument("-dsd", dest="dsd", type=float, default=1.0)
        parser.add_argument("-ss", dest="ss", type=float, default=1.0)
        parser.add_argument("-g", dest="g", type=str, default="")

        return parser

    def reset_experiment(self):
        self.experiment_name = "split_100/-n {}: -t {} -i {} -a {}".format(
            self.name, self.t, self.i, self.a_quant
        )
        self.experiment_name += "-rlw {} samew {}".format(
            str(self.rl_weight), str(self.same_weight)
        )
        self.experiment_name += " -s {} b0 {} tp {} ip {}".format(
            str(self.s), str(self.b0), str(self.tp), str(self.ip)
        )
        self.experiment_name += "-dds {} -dsd {} -ss {}".format(
            str(self.dds), str(self.dsd), str(self.ss)
        )

        print(self.experiment_name)

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

    def choose_operators(self):
        if self.g != "":
            self.G = GENERATORS[self.g]
            self.T = lambda a, b: a + b
            self.I = lambda a, c: torch.clamp(c - a, min=0)
            self.A_quant = lambda a: -torch.sum(a)
        else:
            self.G = lambda x: x
            self.T = CONJUNCTS[self.t]
            self.I = IMPLICATIONS[self.i]
            self.A_quant = AGGREGATORS[self.a_quant]


conf = Config()

test_every_x_epoch = 100
eps = 0.000001

p_yager = conf.tp
ap = p_yager

AGGREGATORS = {}
CONJUNCTS = {}
DISJUNCTS = {}
IMPLICATIONS = {}
GENERATORS = {}

DEBUG = False

if DEBUG:
    torch.autograd.set_detect_anomaly(True)


def debug_id(a):
    # print(a.shape)
    return a


def _agg_np(a):
    a = a.reshape(-1)
    a_sorted = torch.sort(a)[0]
    if a_sorted[0] + a_sorted[1] > 1.0:
        return a_sorted[0]
    return 0.0


# Aggregators
AGGREGATORS["sum"] = lambda a: torch.sum(a)
AGGREGATORS["cross_entropy"] = lambda a: torch.mean(debug_id(torch.log(a + eps)))
AGGREGATORS["mean"] = lambda a: debug_id(torch.mean(a) - 1.0)
AGGREGATORS["hmean"] = lambda a: torch.sqrt(torch.mean(a ** 2.0) + eps)
AGGREGATORS["log_sigmoid"] = lambda a: F.logsigmoid(
    pa * (torch.sum(a) - a.size()[0] + 1.0 + pb0)
)
AGGREGATORS["RMSE"] = lambda a: 1.0 - torch.sqrt(
    torch.mean((1 - a) ** 2.0) + eps
)  # Unclamped Yager/sum of squares
# Adding random noise to make sure it arbitrarily chooses an argument when there are multiple lowest inputs
AGGREGATORS["min"] = lambda a: torch.min(a + torch.randn_like(a) / 1000)
AGGREGATORS["LK"] = lambda a: torch.clamp(torch.sum(a) - a.size()[0] + 1.0, min=0.0)
AGGREGATORS["T"] = (
    lambda a: 2.0 / math.pi * torch.asin(torch.prod(torch.sin(a * math.pi / 2)))
)
AGGREGATORS["H"] = lambda a: 1.0 / (1.0 + torch.sum((1.0 - a) / (a + eps)))
AGGREGATORS["Y"] = lambda a: 1.0 - (torch.sum((1.0 - a) ** ap) + eps) ** (1.0 / ap)
AGGREGATORS["Np"] = _agg_np


def upper_contra(I):
    def _implication(a, c):
        return torch.max(I(a, c), I(1 - c, 1 - a))

    return _implication


def lower_contra(I):
    def _implication(a, c):
        return torch.min(I(a, c), I(1 - c, 1 - a))

    return _implication


def r_impl(I):
    def _r_impl(a, c):
        r = torch.ones_like(a)
        a2 = a[a > c]
        c2 = c[a > c]
        r[a > c] = I(a2, c2)
        if DEBUG and (r != r).any():
            print("nan i")
        return r

    return _r_impl


# Normal product norm
CONJUNCTS["P"] = lambda a, b: a * b
DISJUNCTS["P"] = lambda a, b: a + b - a * b

IMPLICATIONS["RC"] = lambda a, b: DISJUNCTS["P"](1 - a, b)
IMPLICATIONS["quad"] = lambda a, c: DISJUNCTS["P"](1 - a * a, 2 * c - c * c)

# Godel norm
CONJUNCTS["G"] = lambda a, b: torch.min(a, b)
DISJUNCTS["G"] = lambda a, b: torch.max(a, b)
IMPLICATIONS["KD"] = lambda a, b: torch.max(1 - a, b)

IMPLICATIONS["G"] = r_impl(lambda a, c: c)
IMPLICATIONS["uG"] = upper_contra(IMPLICATIONS["G"])
IMPLICATIONS["lG"] = lower_contra(IMPLICATIONS["G"])


# Nilpotent norm
def _npmin(a, b):
    r = torch.zeros_like(a)
    c = a + b <= 1
    a2 = a[c]
    b2 = b.expand(a.size())[c]
    r[c] = torch.min(a2, b2)
    return r


CONJUNCTS["Np"] = lambda a, b: _npmin(a, b)
IMPLICATIONS["F"] = r_impl(lambda a, c: torch.max(1 - a, c))

# # Sigmoid norm
# pa = 6
# pb0 = -0.5
# T_S = lambda a, b: F.sigmoid(pa * (a + b - 1 + pb0))
# S_S = lambda a, b: F.sigmoid(pa * (a + b + pb0))
# I_S = lambda a, c: F.sigmoid(pa * (1-a + c + pb0))

# Yager norm
tp = p_yager
ip = conf.ip
CONJUNCTS["Y"] = lambda a, b: torch.clamp(
    1 - ((1 - a) ** tp + (1 - b) ** tp + eps) ** (1 / tp), min=0
)
DISJUNCTS["Y"] = lambda a, b: torch.clamp((a ** tp + b ** tp + eps) ** (1 / tp), max=1)

IMPLICATIONS["Y"] = lambda a, b: torch.clamp(
    ((1 - a) ** ip + b ** ip + eps) ** (1 / ip), max=1
)
IMPLICATIONS["RMSE"] = lambda a, b: (1 / 2 * ((1 - a) ** ip + b ** ip + eps)) ** (
    1 / ip
)
IMPLICATIONS["RY"] = r_impl(
    lambda a, c: 1 - ((1 - c + eps) ** ip - (1 - a + eps) ** ip + eps) ** (1 / ip)
)

CONJUNCTS["RMSE"] = lambda a, b: 1 - (
    1 / 2 * ((1 - a) ** tp + (1 - b) ** tp) + eps
) ** (1 / tp)

# Luk implications
CONJUNCTS["LK"] = lambda a, b: torch.clamp(a + b - 1, min=0)
DISJUNCTS["LK"] = lambda a, b: torch.clamp(a + b, max=1)
IMPLICATIONS["LK"] = lambda a, c: torch.clamp(1 - a + c, max=1)

# Trigonometric
CONJUNCTS["T"] = (
    lambda a, b: 2
    / math.pi
    * torch.asin(eps + torch.sin(a * math.pi / 2) * torch.sin(b * math.pi / 2))
)
DISJUNCTS["T"] = (
    lambda a, b: 2
    / math.pi
    * torch.acos(eps + torch.cos(a * math.pi / 2) * torch.cos(b * math.pi / 2))
)
IMPLICATIONS["T"] = lambda a, c: DISJUNCTS["T"](1 - a, c)

# Hamacher
v = 0
CONJUNCTS["H"] = lambda a, b: a * b / (eps + v + (1 - v) * (a + b - a * b))
DISJUNCTS["H"] = lambda a, b: (1 - (1 - v) * a * b) / (eps + 1 - (1 - v) * a * b)
IMPLICATIONS["H"] = lambda a, c: DISJUNCTS["H"](1 - a, c)
# Sigmoidal
pa = conf.s
pb0 = conf.b0


# General sigmoid
def sigmoid(f):
    f0 = 0  # f(0, 0)
    f1 = 1  # f(1, 1)
    y1 = math.exp(-pa * (f1 + pb0)) + 1
    y2 = math.exp(-pa * (f0 + pb0)) + 1
    c = y1 / (y2 - y1)

    def _sigmoid(a, b):
        # This is a bug: Doesnt work with f's that have a complete reach. If you use this for implications, it starts
        # acting up as for implications f(0, 0) and f(1, 1) are both 1
        if DEBUG:
            if (a != a).any():
                print("nan a")
            if (b != b).any():
                print("nan b")

        r = c * (y2 * torch.sigmoid(pa * (f(a, b) + pb0)) - 1)
        if DEBUG:
            if (r != r).any():
                print("nan")
                import traceback

                traceback.print_stack()
        return r

    return _sigmoid


CONJUNCTS["SP"] = sigmoid(lambda a, b: a * b)
DISJUNCTS["SP"] = sigmoid(lambda a, b: 1 - (1 - a) * (1 - b))
IMPLICATIONS["SP"] = sigmoid(lambda a, b: 1 - a + a * b)
IMPLICATIONS["SLK"] = sigmoid(IMPLICATIONS["LK"])
IMPLICATIONS["SucLK"] = sigmoid(lambda a, b: 1 - a + b)
IMPLICATIONS["SKD"] = sigmoid(lambda a, b: torch.max(1 - a, b))
IMPLICATIONS["SY"] = sigmoid(IMPLICATIONS["Y"])

IMPLICATIONS["GG"] = r_impl(lambda a, c: c / (a + eps))
IMPLICATIONS["uGG"] = upper_contra(IMPLICATIONS["GG"])
IMPLICATIONS["lGG"] = lower_contra(IMPLICATIONS["GG"])


def normalized_rc(a, c):
    t_implication = 1 - a + a * c
    dMP = a / t_implication
    dMT = (1 - c) / t_implication
    tot_dMP = torch.sum(dMP)
    tot_dMT = torch.sum(dMT)
    return mu * c * dMP / tot_dMP + (1 - mu) * (1 - a) * dMT / tot_dMT


# T-norm generators
GENERATORS["P"] = lambda a: -torch.log(a + eps)

# Choice of aggregator
A_clause = lambda a, b, c: (a + b + c) / 3

conf.choose_operators()

mu = 0.25

# If not mentioned, the value of s is 6
lr = 0.01
# Momentum is really low, why?
momentum = 0.5
log_interval = 100

batch_size = 64
test_batch_size = 100
split_unsup_sup = 100
seed = 1
