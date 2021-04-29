import torch
from argparse import ArgumentParser

from dfl.operators.aggregators import createAggregators, createExistentialAggregators
from dfl.operators.implications import createImplications
from dfl.operators.tnorms import createConjunctsDisjuncts
from dfl.operators.util import eps
import uuid


class SameConfig:
    def __init__(
        self, t, i, a_quant, rl_weight=1.0, ap=2.0, tp=1.5, ip=2.0, alg="adam", s=9.0
    ):
        self.t = t
        self.i = i
        self.a_quant = a_quant
        self.rl_weight = rl_weight
        self.tp = tp
        self.ap = ap
        self.s = s
        self.ip = ip
        self.algorithm = alg
        self.lr = 0.01 if alg == "sgd" else 0.001


class Sum9Config:
    def __init__(
        self,
        t,
        s,
        i,
        a_quant,
        e,
        rl_weight=1.0,
        ep=1.5,
        tp=1.5,
        alg="adam",
        lr=None,
        momentum=0.5,
    ):
        self.t = t
        self.s = s
        self.i = i
        self.a_quant = a_quant
        self.e = e
        self.rl_weight = rl_weight
        self.ep = ep
        self.tp = tp
        self.algorithm = alg
        self.lr = lr
        self.momentum = momentum
        if not lr:
            self.lr = 0.01 if alg == "sgd" else 0.001


class Config(object):
    def __init__(self):
        self.problem = None
        self.t = None
        self.tco = None
        self.i = None
        self.a_quant = None
        self.exists = None
        self.rl_weight = None
        self.same_weight = None
        self.epochs = None
        self.name = None
        self.tp = None
        self.ip = None
        self.ep = None
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
        self.algorithm = None
        self.data_dir = None
        self.out_dir = None
        self.multiconfig = None

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)
        self.epochs += 1

        self.reset_experiment()

    def setup_parser(self):
        parser = ArgumentParser(description="training code")

        parser.add_argument(
            "-p", dest="problem", type=str, default="same", help="same or sum9"
        )
        parser.add_argument("-t", dest="t", type=str, default="Y")
        parser.add_argument("-tco", dest="tco", type=str, default="P")
        parser.add_argument("-i", dest="i", type=str, default="SP")
        parser.add_argument("-a", dest="a_quant", type=str, default="cross_entropy")
        parser.add_argument("-e", dest="exists", type=str, default="max")
        parser.add_argument("-rlw", dest="rl_weight", type=float, default=1.0)
        parser.add_argument("-samew", dest="same_weight", type=float, default=1.0)
        parser.add_argument("-epochs", dest="epochs", type=int, default=7000)
        parser.add_argument("-n", dest="name", type=str, default="")
        parser.add_argument("-tp", dest="tp", type=float, default=2.0)
        parser.add_argument("-ip", dest="ip", type=float, default=0.5)
        parser.add_argument("-ep", dest="ep", type=float, default=1.5)
        parser.add_argument("-s", dest="s", type=float, default=9.0)
        parser.add_argument("-b0", dest="b0", type=float, default=-0.5)
        parser.add_argument("-dds", dest="dds", type=float, default=1.0)
        parser.add_argument("-dsd", dest="dsd", type=float, default=1.0)
        parser.add_argument("-ss", dest="ss", type=float, default=1.0)
        parser.add_argument("-g", dest="g", type=str, default="")
        parser.add_argument("-lr", dest="lr", type=float, default=0.001)
        parser.add_argument("-m", dest="momentum", type=float, default=0.5)
        parser.add_argument("-alg", dest="algorithm", type=str, default="adam")
        parser.add_argument("-datadir", dest="data_dir", type=str, default="../data")
        parser.add_argument("-outdir", dest="out_dir", type=str, default="tb_runs")
        parser.add_argument(
            "-l",
            dest="log_level",
            type=str,
            default="normal",
            help="Tensorboard log level in {normal, all}",
        )
        parser.add_argument(
            "-multiconfig", dest="multiconfig", type=str, default="random"
        )

        return parser

    def reset_to(self, baseConfig: SameConfig):
        self.t = baseConfig.t
        self.i = baseConfig.i
        self.a_quant = baseConfig.a_quant
        self.rl_weight = baseConfig.rl_weight
        self.ip = baseConfig.ip
        self.tp = baseConfig.tp
        self.ep = baseConfig.ap
        self.s = baseConfig.s
        self.lr = baseConfig.lr
        self.algorithm = baseConfig.algorithm
        self.problem = "same"
        self.choose_operators()
        self.reset_experiment()

    def reset_to_sum9(self, baseConfig: Sum9Config):
        self.t = baseConfig.t
        self.i = baseConfig.i
        self.tco = baseConfig.s
        self.a_quant = baseConfig.a_quant
        self.exists = baseConfig.e
        self.rl_weight = baseConfig.rl_weight
        self.ip = baseConfig.tp
        self.tp = baseConfig.tp
        self.ep = baseConfig.ep
        self.lr = baseConfig.lr
        self.momentum = baseConfig.momentum
        self.algorithm = baseConfig.algorithm
        self.problem = "sum9"
        self.choose_operators()
        self.reset_experiment()

    def reset_experiment(self):
        self.experiment_name = "split_100/-n {}: -p {} -t {} -tco {} -i {} -a {} -e {} ".format(
            self.name, self.problem, self.t, self.tco, self.i, self.a_quant, self.exists
        )
        self.experiment_name += "-lr {} -mom {} -rlw {} samew {}".format(
            str(self.lr), str(self.momentum), str(self.rl_weight), str(self.same_weight)
        )
        self.experiment_name += " -s {} b0 {} tp {} ip {} ep {}".format(
            str(self.s), str(self.b0), str(self.tp), str(self.ip), str(self.ep)
        )
        self.experiment_name += "-dds {} -dsd {} -ss {} -alg {} ".format(
            str(self.dds), str(self.dsd), str(self.ss), self.algorithm
        )
        self.experiment_name += uuid.uuid4().hex

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
            self.S = DISJUNCTS[self.tco]
            self.I = IMPLICATIONS[self.i]
            self.A_quant = AGGREGATORS[self.a_quant]
            self.E = EXISTS[self.exists]


DEBUG = False
conf = Config()

conf.debug = DEBUG

test_every_x_epoch = 100

p_yager = conf.tp
ap = p_yager


GENERATORS = {}
CONJUNCTS, DISJUNCTS = createConjunctsDisjuncts(conf)
AGGREGATORS = createAggregators(conf)
EXISTS = createExistentialAggregators(conf)
IMPLICATIONS = createImplications(conf, DISJUNCTS)


if DEBUG:
    torch.autograd.set_detect_anomaly(True)


# T-norm generators
GENERATORS["P"] = lambda a: -torch.log(a + eps)

# Choice of aggregator
A_clause = lambda a, b, c: (a + b + c) / 3
A_clause_sum9 = lambda a, b: (a + b) / 2

conf.choose_operators()

log_interval = 100

batch_size = 64
test_batch_size = 100
split_unsup_sup = 100
seed = 2
dataset_seed = 1
