import torch
from argparse import ArgumentParser

from dfl.operators.aggregators import createAggregators
from dfl.operators.implications import createImplications
from dfl.operators.tnorms import createConjunctsDisjuncts
from dfl.operators.util import eps


class BaseConfig:
    def __init__(self, t, i, a_quant, rl_weight=1.0, p=2.0):
        self.t = t
        self.i = i
        self.a_quant = a_quant
        self.rl_weight = rl_weight
        self.p = p


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
        self.lr = None
        self.momentum = None
        self.log_level = None

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
        parser.add_argument("-epochs", dest="epochs", type=int, default=5000)
        parser.add_argument("-n", dest="name", type=str, default="")
        parser.add_argument("-tp", dest="tp", type=float, default=2.0)
        parser.add_argument("-ip", dest="ip", type=float, default=0.5)
        parser.add_argument("-s", dest="s", type=float, default=9.0)
        parser.add_argument("-b0", dest="b0", type=float, default=-0.5)
        parser.add_argument("-dds", dest="dds", type=float, default=1.0)
        parser.add_argument("-dsd", dest="dsd", type=float, default=1.0)
        parser.add_argument("-ss", dest="ss", type=float, default=1.0)
        parser.add_argument("-g", dest="g", type=str, default="")
        parser.add_argument("-lr", dest="lr", type=float, default=0.01)
        parser.add_argument("-m", dest="momentum", type=float, default=0.5)
        parser.add_argument(
            "-l",
            dest="log_level",
            type=str,
            default="normal",
            help="Tensorboard log level in {normal, all}",
        )

        return parser

    def reset_to(self, baseConfig: BaseConfig):
        self.t = baseConfig.t
        self.i = baseConfig.i
        self.a_quant = baseConfig.a_quant
        self.rl_weight = baseConfig.rl_weight
        self.ip = baseConfig.p
        self.tp = baseConfig.p
        self.choose_operators()
        self.reset_experiment()

    def reset_experiment(self):
        print(self.t)
        print(self.i)
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
        for x, y in vars(self).items():
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


DEBUG = False
conf = Config()

conf.debug = DEBUG

test_every_x_epoch = 100

p_yager = conf.tp
ap = p_yager


GENERATORS = {}
CONJUNCTS, DISJUNCTS = createConjunctsDisjuncts(conf)
AGGREGATORS = createAggregators(conf)
IMPLICATIONS = createImplications(conf, DISJUNCTS)


if DEBUG:
    torch.autograd.set_detect_anomaly(True)


# T-norm generators
GENERATORS["P"] = lambda a: -torch.log(a + eps)

# Choice of aggregator
A_clause = lambda a, b, c: (a + b + c) / 3

conf.choose_operators()

algorithm = "sgd"
log_interval = 100

batch_size = 64
test_batch_size = 100
split_unsup_sup = 100
seed = 1
