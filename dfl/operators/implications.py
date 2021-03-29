import torch
from .util import eps, sigmoidal


def upper_contra(I):
    def _implication(a, c):
        return torch.max(I(a, c), I(1 - c, 1 - a))

    return _implication


def lower_contra(I):
    def _implication(a, c):
        return torch.min(I(a, c), I(1 - c, 1 - a))

    return _implication


def r_impl(I, DEBUG):
    def _r_impl(a, c):
        r = torch.ones_like(a)
        a2 = a[a > c]
        c2 = c[a > c]
        r[a > c] = I(a2, c2)
        if DEBUG and (r != r).any():
            print("nan i")
        return r

    return _r_impl


# I forgot what the idea was of this function...
# Maybe its what I attempted in the NeSY paper
def normalized_rc(a, c):
    mu = 0.25
    t_implication = 1 - a + a * c
    dMP = a / t_implication
    dMT = (1 - c) / t_implication
    tot_dMP = torch.sum(dMP)
    tot_dMT = torch.sum(dMT)
    return mu * c * dMP / tot_dMP + (1 - mu) * (1 - a) * dMT / tot_dMT


def createImplications(conf, DISJUNCTS):
    IMPLICATIONS = {}
    # Gödel based
    IMPLICATIONS["KD"] = lambda a, b: torch.max(1 - a, b)

    IMPLICATIONS["G"] = r_impl(lambda a, c: c, conf.debug)
    IMPLICATIONS["uG"] = upper_contra(IMPLICATIONS["G"])
    IMPLICATIONS["lG"] = lower_contra(IMPLICATIONS["G"])

    # Product based
    IMPLICATIONS["RC"] = lambda a, b: DISJUNCTS["P"](1 - a, b)
    IMPLICATIONS["quad"] = lambda a, c: DISJUNCTS["P"](1 - a * a, 2 * c - c * c)

    IMPLICATIONS["GG"] = r_impl(lambda a, c: c / (a + eps), conf.debug)
    IMPLICATIONS["uGG"] = upper_contra(IMPLICATIONS["GG"])
    IMPLICATIONS["lGG"] = lower_contra(IMPLICATIONS["GG"])

    # Nilpotent Minimum (Fodor)
    IMPLICATIONS["F"] = r_impl(lambda a, c: torch.max(1 - a, c), conf.debug)

    # Yager based
    IMPLICATIONS["Y"] = lambda a, b: torch.clamp(
        ((1 - a) ** conf.ip + b ** conf.ip + eps) ** (1 / conf.ip), max=1
    )
    IMPLICATIONS["RMSE"] = lambda a, b: (
        1 / 2 * ((1 - a) ** conf.ip + b ** conf.ip + eps)
    ) ** (1 / conf.ip)
    IMPLICATIONS["RY"] = r_impl(
        lambda a, c: 1
        - ((1 - c + eps) ** conf.ip - (1 - a + eps) ** conf.ip + eps) ** (1 / conf.ip),
        conf.debug,
    )

    # Lukasiewicz
    IMPLICATIONS["LK"] = lambda a, c: torch.clamp(1 - a + c, max=1)

    # Trigonometric
    IMPLICATIONS["T"] = lambda a, c: DISJUNCTS["T"](1 - a, c)

    # Hamacher
    IMPLICATIONS["H"] = lambda a, c: DISJUNCTS["H"](1 - a, c)

    _sigmoidal = lambda f: sigmoidal(f, conf.s, conf.b0, conf.debug)
    # Sigmoidal implications
    IMPLICATIONS["SP"] = _sigmoidal(lambda a, b: 1 - a + a * b)
    IMPLICATIONS["SLK"] = _sigmoidal(IMPLICATIONS["LK"])
    IMPLICATIONS["SucLK"] = _sigmoidal(lambda a, b: 1 - a + b)
    IMPLICATIONS["SKD"] = _sigmoidal(lambda a, b: torch.max(1 - a, b))
    IMPLICATIONS["SY"] = _sigmoidal(IMPLICATIONS["Y"])

    return IMPLICATIONS
