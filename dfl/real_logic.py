import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
import dfl.config as config
from dfl.config import conf
from dfl.misc import one_hot


class RealLogic(nn.Module):
    def __init__(self, A_clause, A_quant, T, I, G, log_level="normal"):
        super(RealLogic, self).__init__()
        self.A_clause = A_clause
        self.A_quant = A_quant
        self.T = T
        self.I = I
        self.G = G
        self.log_level = log_level

    def logic(self, p1, p2, psame, label1, label2, labels_same, writer, step):
        d_global = Counter()

        def _hook(name, truth_a, truth_c, is_ant):
            self.max_amt_backprop += 1
            log_level = self.log_level

            def hook(grad):
                grad = -grad

                t_name = "/ant/" if is_ant else "/cons/"
                truth_sub_f = truth_a if is_ant else truth_c
                abs_grad = torch.abs(grad)
                if config.DEBUG and torch.sum(abs_grad) > 2000:
                    print("--------------------------------------------")
                    print(name, torch.max(abs_grad))
                    print(name, torch.sum(abs_grad))
                    print(abs_grad.size())
                    print(abs_grad > 0)
                    print(name, abs_grad)
                    print(I1)
                    print(I2)
                    print(I3)

                tot_grad = torch.sum(grad)
                tot_abs_grad = torch.sum(abs_grad)

                g_tag = "global_grad" + t_name
                d_global[g_tag + "grad"] += tot_grad
                d_global[g_tag + "absgrad"] += tot_abs_grad

                # Correct update: The gradient is going in the right direction
                corr_update = (grad > 0) == truth_sub_f
                # Correct reasoning: The gradient is going in the right direction, and this also follows
                # the actual label. Ie, going in the right direction is done using valid logical reasoning
                corr_reason = corr_update & (truth_a == truth_c)
                # print('-----------------------------------')
                # print(tag + 'corr_update', corr_update)
                # print('truth_sub_f', truth_sub_f)
                # print('posgrad', grad>0)
                # print('labels', labels)
                # print(tag + 'absgrad', abs_grad)

                tot_corr_up = torch.sum(abs_grad[corr_update])
                d_global[g_tag + "correct_update"] += tot_corr_up
                tot_corr_reas = torch.sum(abs_grad[corr_reason])
                d_global[g_tag + "correct_reason"] += tot_corr_reas

                if log_level == "all":
                    tag = name + t_name
                    # Write local gradients
                    writer.add_scalar(tag + "grad", tot_grad, step)
                    writer.add_scalar(
                        tag + "correct_update", tot_corr_up / tot_abs_grad, step
                    )

                    writer.add_scalar(
                        tag + "correct_reason", tot_corr_reas / tot_abs_grad, step
                    )

                self.amt_backprop += 1
                if (
                    self.amt_backprop == self.max_amt_backprop
                    or conf.i == "G"  # Edge case: G only has a cons gradient
                    and self.amt_backprop == self.max_amt_backprop / 2
                ):
                    # Commit the global grad.
                    for s in ["/ant/", "/cons/"]:
                        _tag = "global_grad" + s
                        if d_global[_tag + "absgrad"] != 0:
                            writer.add_scalar(
                                _tag + "correct_update",
                                d_global[_tag + "correct_update"]
                                / d_global[_tag + "absgrad"],
                                step,
                            )
                            writer.add_scalar(
                                _tag + "correct_reason",
                                d_global[_tag + "correct_reason"]
                                / d_global[_tag + "absgrad"],
                                step,
                            )
                            writer.add_scalar(
                                _tag + "grad", d_global[_tag + "grad"], step
                            )
                    writer.add_scalar(
                        "global_grad/ratio",
                        d_global["global_grad/cons/absgrad"]
                        / (
                            d_global["global_grad/cons/absgrad"]
                            + d_global["global_grad/ant/absgrad"]
                        ),
                        step,
                    )

            return hook

        def same_hook(grad):
            grad = -grad
            writer.add_scalar("same/grad", torch.sum(grad), step)
            correct = labels_same == (grad > 0)
            writer.add_scalar(
                "same/precision", torch.sum(grad[correct]) / torch.sum(grad), step
            )

        # RL computation of digit(x) & digit(y) -> same(x, y)
        A1 = self.T(p1, p2)
        C1 = psame.repeat(
            1, 10
        )  # Make sure we copy the tensor so that we add a separate node for the consequent for backprop
        I1 = self.I(A1, C1)
        rl1 = self.A_quant(I1) * conf.dds

        # RL computation of digit(x) & same(x, y) -> digit(y)
        A2 = self.T(p1, psame)
        C2 = p2 + 0
        I2 = self.I(A2, C2)
        rl2 = self.A_quant(I2) * conf.dsd

        n = int(np.sqrt(psame.size()[0]))

        # RL computation of same(x, y) -> same(y, x)
        A3 = psame.squeeze(1) + 0
        C3 = psame.squeeze(1).view(n, n).transpose(0, 1).contiguous().view(n * n)
        I3 = self.I(A3, C3)
        rl3 = self.A_quant(I3) * conf.ss

        # RL computation of unique
        #         rl4 = torch.zeros(n)
        #         for i in range(10):
        #             prod = torch.ones(n)
        #             for j in range(10):
        #                 if i == j:
        #                     self.T(prod, )

        # RL computation of same(x, x)
        #         rl4 = self.A_quant(psum9.view(n, n).diag())

        if writer:
            if self.log_level == "verbose":
                writer.add_scalar("dig_dig_then_same/sat", rl1, step)
                writer.add_scalar("dig_same_then_dig/sat", rl2, step)
                writer.add_scalar("samexy_then_sameyx/sat", rl3, step)
                # writer.add_scalar('samexx/sat', rl4, step)

                writer.add_scalar("same/avg_val", torch.mean(psame), step)
                psame.register_hook(same_hook)

            one_hot1 = one_hot(p1, label1)

            one_hot2 = one_hot(p2, label2)

            labels_same = labels_same.unsqueeze(1)
            labels_same_t = (
                labels_same.view(n, n).transpose(0, 1).contiguous().view(n * n)
            )

            ta1 = one_hot1 * one_hot2
            A1.register_hook(_hook("dig_dig_then_same", ta1, labels_same, True))
            C1.register_hook(_hook("dig_dig_then_same", ta1, labels_same, False))

            ta2 = one_hot1 * labels_same
            A2.register_hook(_hook("dig_same_then_dig", ta2, one_hot2, True))
            C2.register_hook(_hook("dig_same_then_dig", ta2, one_hot2, False))

            _labels_same = labels_same.squeeze(1)
            A3.register_hook(
                _hook("samexy_then_sameyx", _labels_same, labels_same_t, True)
            )
            C3.register_hook(
                _hook("samexy_then_sameyx", _labels_same, labels_same_t, False)
            )
            # I3.register_hook(_hook('samexy_then_sameyx', _labels_same, labels_same_t, True))
        return -self.A_clause(rl1, rl2, rl3)

    def forward(self, result, writer, step):
        logits1 = result["logits_1"]
        logits2 = result["logits_2"]
        logits_rel = result["logits_same"]
        p1 = self.G(F.softmax(logits1, dim=-1))
        p2 = self.G(F.softmax(logits2, dim=-1))
        psame = self.G(torch.sigmoid(logits_rel))

        label1 = result["labels_1"]
        label2 = result["labels_2"]
        labels_same = result["labels_same"]

        # Keep track of when to update the global statistics: This is after the backward hooks on ant + cons for the three formulas
        self.amt_backprop = 0
        self.max_amt_backprop = 0

        return self.logic(p1, p2, psame, label1, label2, labels_same, writer, step)
        # Computes the backward hook for monitoring


class Sum9Logic(RealLogic):
    def __init__(self, A_clause, A_quant, E, T, S, I, G, log_level="normal"):
        super(Sum9Logic, self).__init__(A_clause, A_quant, T, I, G, log_level)
        self.E = E
        self.S = S

    def logic(self, p1, p2, psum9, label1, label2, labels_sum9, writer, step):
        d_global = Counter()

        def _hook(name, truth_a, truth_c, is_ant):
            self.max_amt_backprop += 1
            log_level = self.log_level

            def hook(grad):
                grad = -grad
                t_name = "/ant/" if is_ant else "/cons/"
                truth_sub_f = truth_a if is_ant else truth_c
                abs_grad = torch.abs(grad)
                if config.DEBUG and torch.sum(abs_grad) > 2000:
                    print("--------------------------------------------")
                    print(name, torch.max(abs_grad))
                    print(name, torch.sum(abs_grad))
                    print(abs_grad.size())
                    print(abs_grad > 0)
                    print(name, abs_grad)
                    print(I2)

                tot_grad = torch.sum(grad)
                tot_abs_grad = torch.sum(abs_grad)

                g_tag = "global_grad" + t_name
                d_global[g_tag + "grad"] += tot_grad
                d_global[g_tag + "absgrad"] += tot_abs_grad

                # Correct update: The gradient is going in the right direction
                corr_update = (grad > 0) == truth_sub_f
                # Correct reasoning: The gradient is going in the right direction, and this also follows
                # the actual label. Ie, going in the right direction is done using valid logical reasoning
                corr_reason = corr_update & (truth_a == truth_c)
                # print('-----------------------------------')
                # print(tag + 'corr_update', corr_update)
                # print('truth_sub_f', truth_sub_f)
                # print('posgrad', grad>0)
                # print('labels', labels)
                # print(tag + 'absgrad', abs_grad)

                tot_corr_up = torch.sum(abs_grad[corr_update])
                d_global[g_tag + "correct_update"] += tot_corr_up
                tot_corr_reas = torch.sum(abs_grad[corr_reason])
                d_global[g_tag + "correct_reason"] += tot_corr_reas

                if log_level == "all":
                    tag = name + t_name
                    # Write local gradients
                    writer.add_scalar(tag + "grad", tot_grad, step)
                    writer.add_scalar(
                        tag + "correct_update", tot_corr_up / tot_abs_grad, step
                    )

                    writer.add_scalar(
                        tag + "correct_reason", tot_corr_reas / tot_abs_grad, step
                    )

                self.amt_backprop += 1
                if (
                    self.amt_backprop == self.max_amt_backprop
                    or conf.i == "G"  # Edge case: G only has a cons gradient
                    and self.amt_backprop == self.max_amt_backprop / 2
                ):
                    # Commit the global grad.
                    for s in ["/ant/", "/cons/"]:
                        _tag = "global_grad" + s
                        if d_global[_tag + "absgrad"] != 0:
                            writer.add_scalar(
                                _tag + "correct_update",
                                d_global[_tag + "correct_update"]
                                / d_global[_tag + "absgrad"],
                                step,
                            )
                            writer.add_scalar(
                                _tag + "correct_reason",
                                d_global[_tag + "correct_reason"]
                                / d_global[_tag + "absgrad"],
                                step,
                            )
                            writer.add_scalar(
                                _tag + "grad", d_global[_tag + "grad"], step
                            )
                    writer.add_scalar(
                        "global_grad/ratio",
                        d_global["global_grad/cons/absgrad"]
                        / (
                            d_global["global_grad/cons/absgrad"]
                            + d_global["global_grad/ant/absgrad"]
                        ),
                        step,
                    )

            return hook

        def same_hook(grad):
            grad = -grad
            writer.add_scalar("same/grad", torch.sum(grad), step)
            correct = labels_sum9 == (grad > 0)
            writer.add_scalar(
                "same/precision", torch.sum(grad[correct]) / torch.sum(grad), step
            )

        n = int(np.sqrt(psum9.size()[0]))
        # RL computation of \forall x \exists y Sum9(x, y)
        # This formula can be wrong in an unfortunate minibatch.
        # Probability of wrong can be computed using prob_sum9_wrong.py
        # For b=64, this probability is 0.01176195
        asMatrix = psum9.view(n, n)
        Exists = self.E(asMatrix)
        rl1 = self.A_quant(Exists)

        # RL computation of digit(x) & same(x, y) -> digit(y)
        A2 = psum9.squeeze(1)
        disjunct = torch.zeros_like(A2)
        for i in range(10):
            disjunct = self.S(disjunct, self.T(p1[:, i], p2[:, 9 - i]))
        C2 = disjunct
        I2 = self.I(A2, C2)
        rl2 = self.A_quant(I2)

        if writer:
            if self.log_level == "verbose":
                writer.add_scalar("dig_dig_then_same/sat", rl1, step)
                writer.add_scalar("dig_same_then_dig/sat", rl2, step)
                # writer.add_scalar('samexx/sat', rl4, step)

                writer.add_scalar("same/avg_val", torch.mean(psum9), step)
                psum9.register_hook(same_hook)

            # labels_sum9 = labels_sum9.unsqueeze(1)

            # ta1 = one_hot1 * one_hot2
            # A1.register_hook(_hook("dig_dig_then_same", ta1, labels_sum9, True))
            # C1.register_hook(_hook("dig_dig_then_same", ta1, labels_sum9, False))

            A2.register_hook(_hook("sum9_then_plus", labels_sum9, labels_sum9, True))
            C2.register_hook(_hook("sum9_then_plus", labels_sum9, labels_sum9, False))

        return -self.A_clause(rl1, rl2)
