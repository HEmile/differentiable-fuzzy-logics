import torch
from .util import eps
import torch.nn.functional as F
import math


def _agg_np(a):
    a = a.reshape(-1)
    a_sorted = torch.sort(a)[0]
    if a_sorted[0] + a_sorted[1] > 1.0:
        return a_sorted[0]
    return 0.0


def createAggregators(conf):
    AGGREGATORS = {}
    # Aggregators
    AGGREGATORS["sum"] = lambda a: torch.sum(a)
    AGGREGATORS["cross_entropy"] = lambda a: torch.mean(torch.log(a + eps))
    AGGREGATORS["mean"] = lambda a: torch.mean(a) - 1.0
    AGGREGATORS["hmean"] = lambda a: torch.sqrt(torch.mean(a ** 2.0) + eps)
    AGGREGATORS["log_sigmoid"] = lambda a: F.logsigmoid(
        conf.s * (torch.sum(a) - a.size()[0] + 1.0 + conf.b0)
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
    AGGREGATORS["Y"] = lambda a: 1.0 - (torch.sum((1.0 - a) ** conf.tp) + eps) ** (
        1.0 / conf.tp
    )
    AGGREGATORS["Np"] = _agg_np
    return AGGREGATORS
