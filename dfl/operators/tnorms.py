import torch
from dfl.operators.util import eps, sigmoidal
import math

# Nilpotent norm
def _npmin(a, b):
    c = (a + b <= 1).float()
    return (1 - c) * torch.min(a, b)


def _npmax(a, b):
    c = (a + b >= 1).float()
    return c + (1.0 - c) * torch.max(a, b)


def createConjunctsDisjuncts(conf):
    CONJUNCTS = {}
    DISJUNCTS = {}
    # Normal product norm
    CONJUNCTS["P"] = lambda a, b: a * b
    DISJUNCTS["P"] = lambda a, b: a + b - a * b

    # Godel norm
    CONJUNCTS["G"] = lambda a, b: torch.min(a, b)
    DISJUNCTS["G"] = lambda a, b: torch.max(a, b)

    CONJUNCTS["Np"] = _npmin
    DISJUNCTS["Np"] = _npmax
    # Yager based
    CONJUNCTS["Y"] = lambda a, b: torch.clamp(
        1
        - ((1 - a + eps) ** conf.tp + (1 - b + eps) ** conf.tp + eps) ** (1 / conf.tp),
        min=0,
    )
    DISJUNCTS["Y"] = lambda a, b: torch.clamp(
        ((a + eps) ** conf.tp + (b + eps) ** conf.tp + eps) ** (1 / conf.tp), max=1
    )

    CONJUNCTS["RMSE"] = lambda a, b: 1 - (
        1 / 2 * ((1 - a) ** conf.tp + (1 - b) ** conf.tp) + eps
    ) ** (1 / conf.tp)
    # Luk implications
    CONJUNCTS["LK"] = lambda a, b: torch.clamp(a + b - 1, min=0)
    DISJUNCTS["LK"] = lambda a, b: torch.clamp(a + b, max=1)

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
    # Hamacher
    v = 0
    CONJUNCTS["H"] = lambda a, b: a * b / (eps + v + (1 - v) * (a + b - a * b))
    DISJUNCTS["H"] = lambda a, b: (1 - (1 - v) * a * b) / (eps + 1 - (1 - v) * a * b)

    # Sigmoidal
    CONJUNCTS["SP"] = sigmoidal(lambda a, b: a * b, conf.s, conf.b0, conf.debug)
    DISJUNCTS["SP"] = sigmoidal(
        lambda a, b: 1 - (1 - a) * (1 - b), conf.s, conf.b0, conf.debug
    )

    return CONJUNCTS, DISJUNCTS
