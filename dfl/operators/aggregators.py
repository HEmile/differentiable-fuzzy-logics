import torch
from dfl.operators.util import eps, sigmoidal
import torch.nn.functional as F
import math


def _agg_np(a):
    a = a.reshape(-1)
    a_sorted = torch.sort(a)[0]
    if a_sorted[0] + a_sorted[1] > 1.0:
        return a_sorted[0]
    return torch.zeros(())


def _exists_np(a):
    a_sorted = torch.sort(a, dim=-1)[0]
    cond = (a_sorted[:, -1] + a_sorted[:, -2] < 1.0).float()
    return (1.0 - cond) + cond * a_sorted[:, -1]


def createAggregators(conf):
    AGGREGATORS = {}
    # Aggregators
    AGGREGATORS["sum"] = lambda a: torch.sum(a)
    AGGREGATORS["cross_entropy"] = lambda a: torch.mean(torch.log(a + eps))
    AGGREGATORS["mean"] = lambda a: torch.mean(a) - 1.0
    AGGREGATORS["GME"] = lambda a: 1 - (torch.mean((1 - a + eps) ** conf.ep) + eps) ** (
        1.0 / conf.ep
    )
    AGGREGATORS["hmean"] = lambda a: torch.sqrt(torch.mean(a ** 2.0) + eps)
    AGGREGATORS["log_sigmoid"] = lambda a: F.logsigmoid(
        conf.s * (torch.sum(a) - torch.numel(a) + 1.0 + conf.b0)
    )
    AGGREGATORS["RMSE"] = lambda a: 1.0 - torch.sqrt(
        torch.mean((1 - a) ** 2.0) + eps
    )  # Unclamped Yager/sum of squares
    # Adding random noise to make sure it arbitrarily chooses an argument when there are multiple lowest inputs
    AGGREGATORS["min"] = lambda a: torch.min(a + torch.randn_like(a) / 1000)
    AGGREGATORS["LK"] = lambda a: torch.clamp(
        torch.sum(a) - torch.numel(a) + 1.0, min=0.0
    )
    AGGREGATORS["T"] = (
        lambda a: 2.0 / math.pi * torch.asin(torch.prod(torch.sin(a * math.pi / 2)))
    )
    AGGREGATORS["H"] = lambda a: 1.0 / (1.0 + torch.sum((1.0 - a) / (a + eps)))
    AGGREGATORS["Y"] = lambda a: torch.clamp(
        1.0 - (torch.sum((1.0 - a) ** conf.tp) + eps) ** (1.0 / conf.tp), min=0.0
    )
    AGGREGATORS["Np"] = _agg_np
    return AGGREGATORS


def createExistentialAggregators(conf):
    EXISTS = {}

    EXISTS["sum"] = lambda a: torch.sum(a, dim=-1)
    EXISTS["mean"] = lambda a: torch.mean(a, dim=-1)
    EXISTS["hmean"] = lambda a: torch.reciprocal(torch.mean(torch.reciprocal(a)))
    EXISTS["sqmean"] = lambda a: torch.sqrt(torch.mean(a ** 2.0, dim=-1) + eps)
    EXISTS["gmean"] = lambda a: (torch.mean((a + eps) ** conf.ep, dim=-1) + eps) ** (
        1.0 / conf.ep
    )

    EXISTS["max"] = lambda a: torch.max(a + torch.randn_like(a) / 10000, dim=-1).values
    EXISTS["prob_sum"] = lambda a: 1 - torch.prod(1 - a, dim=-1)
    # Numerically stable implementation of prob_sum, should be combined with the mean universal aggregator
    EXISTS["log_prob_sum"] = lambda a: torch.log1p(-torch.prod(1 - a + eps, dim=-1))
    EXISTS["LK"] = lambda a: torch.clamp(torch.sum(a, dim=-1), max=1.0)

    EXISTS["Np"] = _exists_np
    EXISTS["Y"] = lambda a: torch.clamp(
        (torch.sum((a + eps) ** conf.ep, dim=-1) + eps) ** (1.0 / conf.ep), max=1.0
    )

    # Note: This is not symmetric, and actually isn't allowed at all.
    EXISTS["SP"] = lambda a: sigmoidal(EXISTS["prob_sum"], conf.s, conf.b0, conf.debug)

    return EXISTS
