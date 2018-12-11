import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import one_hot, cross_prod
import numpy as np
from neural_config import balance_classes

class NeuralDR(nn.Module):
    def __init__(self, hidden):
        super(NeuralDR, self).__init__()
        # dd->s # Note this is a bi-implication lol
        self.fcdds1 = nn.Linear(3, hidden)
        self.fcdds2 = nn.Linear(hidden, 1)

        #ds->d
        self.fcdsd1 = nn.Linear(3, hidden)
        self.fcdsd2 = nn.Linear(hidden, 1)

        #ss (symmetry)
        self.fcss1 = nn.Linear(2, hidden)
        self.fcss2 = nn.Linear(hidden, 1)

    def set_requires_grad(self, val):
        for p in self.parameters():
            p.requires_grad = val

    def forward(self, result, writer, step, training):
        logits1 = result['logits_1']
        logits2 = result['logits_2']
        logits_same = result['logits_same']

        # Normalize between -1 and 1
        p1 = 2 * F.softmax(logits1, dim=-1) - 1
        p2 = 2 * F.softmax(logits2, dim=-1) - 1
        psame = 2 * torch.sigmoid(logits_same) - 1
        psamer = psame.repeat(1, 10)

        if training:
            # # digit(x) & digit(y) <-> same(x, y)
            # dr1 = self.fcdds1(i1)
            # dr1 = self.fcdds2(dr1)

            # digit(x) & same(x, y) -> digit(y)
            examples, labels = self.dataset_dsd(p1, p2, psamer)
            dr2 = self.fcdsd1(examples)
            dr2 = self.fcdsd2(dr2)
            dr2 = torch.nn.BCEWithLogitsLoss()(dr2.squeeze(), labels)

            # same(x, y) <-> same(y, x)
            examples, labels = self.dataset_ss(psame, result)
            dr3 = self.fcss1(examples)
            dr3 = self.fcss2(dr3)
            dr3 = torch.nn.BCEWithLogitsLoss()(dr3.squeeze(), labels)

            # print(examples[:12], torch.sigmoid(dr2)[:12], labels[:12], examples[-12:], torch.sigmoid(dr2)[-12:], labels[-12:])

            return dr2 + dr3
        else:
            self.set_requires_grad(False)

            positive_examples = torch.stack([p1, psamer, p2], dim=2).view(-1, 3)
            dr2 = self.fcdsd1(positive_examples)
            dr2 = self.fcdsd2(dr2)

            dr2 = torch.nn.BCEWithLogitsLoss()(dr2.squeeze(), dr2.new_ones(positive_examples.size()[0]))

            n = int(np.sqrt(psame.size()[0]))
            a = psame.squeeze(1)
            c = psame.squeeze(1).view(n, n).transpose(0, 1).contiguous().view(n * n)
            positive_examples = torch.stack([a, c], dim=1).view(-1, 2)
            dr3 = self.fcss1(positive_examples)
            dr3 = self.fcss2(dr3)
            dr3 = torch.nn.BCEWithLogitsLoss()(dr3.squeeze(), dr3.new_ones(positive_examples.size()[0]))
            self.set_requires_grad(True)
            return dr2 + dr3

    # digit(x) & same(x, y) -> digit(y)
    def dataset_dsd(self, p1, p2, psame):
        # one_hot1 = one_hot(p1, labels1)
        # one_hot2 = one_hot(p2, labels2)
        #
        # truth_a = one_hot1 * labels_same

        # print(torch.sum(mp)) # Test: 468
        # print(torch.sum(mt)) # Test: 36864
        # print(torch.sum(neutral)) # Test: 3628

        # Create negative examples for this implication.
        # If it is an mp (ie, both are true), negate the consequent, for mt, negate antecedent. If 0, 1, negate both.

        positive_examples = torch.stack([p1, psame, p2], dim=2).view(-1, 3)
        n = positive_examples.size()[0]
        batch_size = 128

        absed_1, absed2, absed_same = torch.abs(p1), torch.abs(p2), torch.abs(psame)
        negative_examples = torch.stack([absed_1, absed_same, -absed2], dim=2).view(-1, 3)

        positive_examples = positive_examples[torch.randperm(n)[:batch_size]]
        negative_examples = negative_examples[torch.randperm(n)[:batch_size]]

        examples = torch.cat([positive_examples, negative_examples]).detach()

        labels = torch.cat([examples.new_ones(batch_size), examples.new_zeros(batch_size)])

        return examples, labels

    def dataset_ss(self, psame, result):
        # same(x, y) -> same(y, x). So, there are exactly batch_size^2 such examples. If same(x, y) is true in the data
        # then by definition same(y, x) is.
        n = int(np.sqrt(psame.size()[0]))
        a = psame.squeeze(1)
        c = psame.squeeze(1).view(n, n).transpose(0, 1).contiguous().view(n * n)
        same_labels = result['labels_same']
        positive_examples = torch.stack([a, c], dim=1).view(-1, 2)
        n = positive_examples.size()[0]
        batch_size = 128

        absa, absc = torch.abs(a), torch.abs(c)
        negative_examples1 = torch.stack([absa, -absc], dim=1).view(-1, 2)
        negative_examples2 = torch.stack([-absa, absc], dim=1).view(-1, 2)
        negative_examples = torch.cat([negative_examples1, negative_examples2])

        mp_positive_examples = positive_examples[same_labels]
        mt_positive_examples = positive_examples[~same_labels]

        positive_examples1 = mp_positive_examples[torch.randint(high=mp_positive_examples.size()[0], size=(64,)).long()]
        positive_examples2 = mt_positive_examples[torch.randperm(mt_positive_examples.size()[0])[:64]]
        negative_examples = negative_examples[torch.randperm(2*n)[:batch_size]]

        examples = torch.cat([positive_examples1, positive_examples2, negative_examples]).detach()

        labels = torch.cat([examples.new_ones(batch_size), examples.new_zeros(batch_size)])

        return examples, labels

    def get_test_graph(self, device, experiment_name, step):
        x = cross_prod(torch.linspace(-1., 1., steps=50)).to(device)
        dr3 = self.fcss1(x)
        dr3 = self.fcss2(dr3)
        outp = torch.sigmoid(dr3).view(50, 50).cpu().numpy()
        np.savetxt('graphs/' + experiment_name + str(step) + '.csv', outp, delimiter=',')

# In multiclass neural dr, we predict not whether the implication is satisfied, but what the correct
# configuration should be.
# Training a multiclass neural dr works as follows: On data we have not done supervised learning over but we do have
# labels for, compute the predicted truth values of the predicates then correct the class prediction by the true
# supervision labels.
# For differentiable reasoning, compute the forward pass, then take the maximum value as the supervision label.
class MultiNeuralDR(nn.Module):
    def __init__(self, hidden):
        super(MultiNeuralDR, self).__init__()
        # dd->s # Note this is a bi-implication lol
        self.fcdds1 = nn.Linear(3, hidden)
        self.fcdds2 = nn.Linear(hidden, 8)

        #ds->d
        self.fcdsd1 = nn.Linear(3, hidden)
        self.fcdsd2 = nn.Linear(hidden, 8)

        #ss (symmetry)
        self.fcss1 = nn.Linear(2, hidden)
        self.fcss2 = nn.Linear(hidden, 4)

    def set_requires_grad(self, val):
        for p in self.parameters():
            p.requires_grad = val

    def forward(self, result, writer, step, training):
        logits1 = result['logits_1']
        logits2 = result['logits_2']
        logits_same = result['logits_same']

        # Normalize between -1 and 1
        p1 = 2 * F.softmax(logits1, dim=-1) - 1
        p2 = 2 * F.softmax(logits2, dim=-1) - 1
        psame = 2 * torch.sigmoid(logits_same) - 1
        psamer = psame.repeat(1, 10)

        if training:
            # # digit(x) & digit(y) <-> same(x, y)
            # dr1 = self.fcdds1(i1)
            # dr1 = self.fcdds2(dr1)

            # digit(x) & same(x, y) -> digit(y)
            examples, labels = self.dataset_dsd(p1, p2, psamer, result)
            dr2 = self.fcdsd1(examples)
            dr2 = self.fcdsd2(dr2)
            dr2 = torch.nn.CrossEntropyLoss()(dr2.squeeze(), labels)

            # same(x, y) <-> same(y, x)
            examples, labels = self.dataset_ss(psame, result)
            dr3 = self.fcss1(examples)
            dr3 = self.fcss2(dr3)
            dr3 = torch.nn.CrossEntropyLoss()(dr3.squeeze(), labels)

            # print(examples[:12], torch.sigmoid(dr2)[:12], labels[:12], examples[-12:], torch.sigmoid(dr2)[-12:], labels[-12:])

            return dr2 + dr3
        else:
            self.set_requires_grad(False)

            positive_examples = torch.stack([p1, psamer, p2], dim=2).view(-1, 3)
            dr2 = self.fcdsd1(positive_examples)
            dr2 = self.fcdsd2(dr2)

            # Feed the highest class as the label
            dr2 = torch.nn.CrossEntropyLoss()(dr2.squeeze(), torch.argmax(dr2, dim=1))

            n = int(np.sqrt(psame.size()[0]))
            a = psame.squeeze(1)
            c = psame.squeeze(1).view(n, n).transpose(0, 1).contiguous().view(n * n)
            positive_examples = torch.stack([a, c], dim=1).view(-1, 2)
            dr3 = self.fcss1(positive_examples)
            dr3 = self.fcss2(dr3)
            dr3 = torch.nn.CrossEntropyLoss()(dr3.squeeze(), torch.argmax(dr3, dim=1))
            self.set_requires_grad(True)
            return dr2 + dr3

    # digit(x) & same(x, y) -> digit(y)
    def dataset_dsd(self, p1, p2, psame, result):
        labels1 = result['labels_1']
        labels2 = result['labels_2']
        labels_same = result['labels_same'].unsqueeze(1)
        one_hot1 = one_hot(p1, labels1)
        one_hot2 = one_hot(p2, labels2)

        # Create the 8 possible classes. Basically, each truth value is a bit.
        labels = one_hot1 + one_hot2 * 2 + labels_same * 4

        positive_examples = torch.stack([p1, psame, p2], dim=2).view(-1, 3)
        n = positive_examples.size()[0]
        batch_size = 128

        indexes = torch.randperm(n)[:batch_size]

        examples = positive_examples[indexes]
        labels = labels.view(-1)[indexes].long()

        return examples, labels

    def dataset_ss(self, psame, result):
        # same(x, y) -> same(y, x). So, there are exactly batch_size^2 such examples. If same(x, y) is true in the data
        # then by definition same(y, x) is.
        n = int(np.sqrt(psame.size()[0]))
        a = psame.squeeze(1)
        c = psame.squeeze(1).view(n, n).transpose(0, 1).contiguous().view(n * n)
        same_labels = result['labels_same']
        positive_examples = torch.stack([a, c], dim=1).view(-1, 2)
        batch_size_per_class = 64

        if balance_classes:
            mp_positive_examples = positive_examples[same_labels]
            mt_positive_examples = positive_examples[~same_labels]

            positive_examples1 = mp_positive_examples[torch.randint(high=mp_positive_examples.size()[0], size=(batch_size_per_class,)).long()]
            positive_examples2 = mt_positive_examples[torch.randperm(mt_positive_examples.size()[0])[:batch_size_per_class]]
            examples = torch.cat([positive_examples1, positive_examples2]).detach()

            labels = torch.cat([examples.new_ones(batch_size_per_class), examples.new_zeros(batch_size_per_class)]).long()
        else:
            indexes = torch.randperm(positive_examples.size()[0])
            examples = positive_examples[indexes]
            labels = same_labels[indexes].long()

        return examples, labels

    def get_test_graph(self, device, experiment_name, step):
        x = cross_prod(torch.linspace(-1., 1., steps=50)).to(device)
        dr3 = self.fcss1(x)
        dr3 = self.fcss2(dr3)
        outp = torch.argmax(dr3, dim=1).view(50, 50).cpu().numpy()
        probs = torch.nn.functional.softmax(dr3, dim=1).view(50, 50, -1).cpu().numpy()
        np.savetxt('graphs/' + experiment_name +'classes'+ str(step) + '.csv', outp, delimiter=',')
        for i in range(probs.shape[-1]):
            np.savetxt('graphs/' + experiment_name + 'class' + str(i) + str(step) + '.csv', probs[:, :, i], delimiter=',')

