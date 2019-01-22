import torch
from misc import add_print_hooks
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, writer, step):
        a1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        a2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(a1)), 2))
        a2 = a2.view(-1, 320)
        embedding = F.relu(self.fc1(a2))
        a3 = F.dropout(embedding, training=self.training)
        a3 = self.fc2(a3)
        if writer != None:
            writer.add_scalar('activations/stdev1', torch.std(a1), step)
            writer.add_scalar('activations/stdev2', torch.std(a2), step)
            writer.add_scalar('activations/stdev3', torch.std(a3), step)
        return a3, embedding

# #The SameNet model takes both labeled and unlabeled data, does a forward pass on the MNIST model, then gets the
# # embeddings of the unlabeled data and concatenates every possible combination of embeddings and does another forward
# # pass to compute the truth value of the predicate Same.
# class SameNet(nn.Module):
#     def __init__(self):
#         super(SameNet, self).__init__()
#         self.detector = Net()
#         # Linear same relation
#         k = 10
#         self.fcrel = nn.Linear(100, k)
#         self.bilinear = nn.Bilinear(50, 50, k)
#         self.lastfc = nn.Linear(k, 1)
#
#     def get_same_logits(self, embedding, labels):
#         # Computes the concatenation of every possible combination of embeddings
#         n, _ = embedding.size()
#         pairs = []
#         x1 = []
#         x2 = []
#         for i in range(n):
#             repeat_i = embedding[i].repeat(n, 1)
#             pairs.append(torch.cat([embedding, repeat_i], 1))
#             x1.append(embedding)
#             x2.append(repeat_i)
#         embed_pairs = torch.cat(pairs)
#         x1 = torch.cat(x1)
#         x2 = torch.cat(x2)
#
#         # Compute the forward pass
#         V = self.fcrel(embed_pairs)
#         W = self.bilinear(x1, x2)
#         logits_same = self.lastfc(torch.tanh(V + W))
#         # logits_same = self.fcrel2(hidden)
#
#         # Pairs the labels to find the array of targets for same
#         labels_same = []
#         for i in range(labels.size()[0]):
#             labels_same.append(labels == labels[i])
#         return logits_same, torch.cat(labels_same)
#
#     def forward(self, x_sup, labels_sup, x_unsup, labels_unsup, writer, step):
#         # Forward pass through digit recognition model
#         logits_sup, e_sup = self.detector(x_sup, writer, step)
#         logits_unsup, e_unsup = self.detector(x_unsup, None, step)
#
#         # Correctly match the labels of the unsupervised pass
#         n, _ = logits_unsup.size()
#         logits_x = logits_unsup.repeat(n, 1)
#         labels_x = labels_unsup.repeat(n)
#         logits_y_l = []
#         labels_y = labels_unsup.new_zeros(n * n)
#         for i in range(n):
#             logits_y_l.append(logits_unsup[i].repeat(n, 1))
#             labels_y[i * n:(i + 1) * n] = labels_unsup[i]
#         logits_y = torch.cat(logits_y_l)
#
#         # Forward pass to same model for both labeled and unlabeled. Reuses the earlier embeddings
#         logits_same_sup, labels_same_sup = self.get_same_logits(e_sup, labels_sup)
#         logits_same_unsup, labels_same_unsup = self.get_same_logits(e_unsup, labels_unsup)
#
#         unsup_labels = labels_x, labels_y, labels_same_unsup
#
#         # Hook to be used by pytorch in the backward pass for debugging and monitoring
#         def hook(grad):
#             g_tot_grad = 0
#             g_truth_grad = 0
#             for i in range(grad.size()[1]):
#                 tag = 'class_grad/' + str(i)
#                 tensor = -grad[:, i]
#                 truth = (labels_unsup == i) == (tensor > 0)
#
#                 tot_grad = torch.sum(torch.abs(tensor))
#                 writer.add_scalar(tag, tot_grad, step)
#                 g_tot_grad += tot_grad
#                 true_grad = torch.sum(torch.abs(tensor[truth]))
#                 g_truth_grad += true_grad
#                 writer.add_scalar(tag + '/precision', true_grad / tot_grad, step)
#             writer.add_scalar('class_grad/global', g_tot_grad, step)
#             writer.add_scalar('class_grad/global_precision', g_truth_grad / g_tot_grad, step)
#
#         if writer:
#             logits_unsup.register_hook(hook)
#             writer.add_scalar('activations/stdev_same_logits', torch.std(logits_same_sup), step)
#             add_print_hooks(logits_same_sup, writer, 'activations/same', step)
#
#         return logits_sup, logits_same_sup.squeeze(
#             1), labels_same_sup, logits_x, logits_y, logits_same_unsup, unsup_labels


#The SameNet model takes both labeled and unlabeled data, does a forward pass on the MNIST model, then gets the
# embeddings of the unlabeled data and concatenates every possible combination of embeddings and does another forward
# pass to compute the truth value of the predicate Same.
class SameNet(nn.Module):
    def __init__(self):
        super(SameNet, self).__init__()
        self.detector = Net()
        # Linear same relation
        k = 10
        self.fcrel = nn.Linear(100, k)
        self.bilinear = nn.Bilinear(50, 50, k)
        self.lastfc = nn.Linear(k, 1)

    def get_same_logits(self, embedding, labels):
        # Computes the concatenation of every possible combination of embeddings
        import time
        time1 = time.time()
        n, _ = embedding.size()
        pairs = []
        x1 = []
        x2 = []
        for i in range(n):
            repeat_i = embedding[i].repeat(n, 1)
            pairs.append(torch.cat([embedding, repeat_i], 1))
            x1.append(embedding)
            x2.append(repeat_i)
        embed_pairs = torch.cat(pairs)
        x1 = torch.cat(x1)
        x2 = torch.cat(x2)

        # Compute the forward pass
        V = self.fcrel(embed_pairs)
        W = self.bilinear(x1, x2)
        logits_same = self.lastfc(torch.tanh(V + W))
        # logits_same = self.fcrel2(hidden)

        # Pairs the labels to find the array of targets for same
        n2 = labels.size()[0]
        bb = labels.new_zeros(10, n2)
        for i in range(10):
            bb[i, :] = labels == i
        labels_same = bb[labels, :].view(-1)

        return logits_same, labels_same.byte()

    def forward(self, x, labels, writer, step):
        # Forward pass through digit recognition model
        logits, e = self.detector(x, writer, step)

        # Correctly match the labels of the unsupervised pass
        n, _ = logits.size()
        logits_x = logits.repeat(n, 1)
        labels_x = labels.repeat(n)
        logits_y_l = []
        labels_y = labels.new_zeros(n * n)
        for i in range(n):
            logits_y_l.append(logits[i].repeat(n, 1))
            labels_y[i * n:(i + 1) * n] = labels[i]
        logits_y = torch.cat(logits_y_l)

        # Forward pass to same model for both labeled and unlabeled. Reuses the earlier embeddings
        logits_same, labels_same = self.get_same_logits(e, labels)

        # Hook to be used by pytorch in the backward pass for debugging and monitoring
        def hook(grad):
            g_tot_grad = 0
            g_truth_grad = 0
            for i in range(grad.size()[1]):
                tag = 'class_grad/' + str(i)
                tensor = -grad[:, i]
                truth = (labels == i) == (tensor > 0)

                tot_grad = torch.sum(torch.abs(tensor))
                writer.add_scalar(tag, tot_grad, step)
                g_tot_grad += tot_grad
                true_grad = torch.sum(torch.abs(tensor[truth]))
                g_truth_grad += true_grad
                writer.add_scalar(tag + '/precision', true_grad / tot_grad, step)
            writer.add_scalar('class_grad/global', g_tot_grad, step)
            writer.add_scalar('class_grad/global_precision', g_truth_grad / g_tot_grad, step)

        if writer:
            logits.register_hook(hook)
            writer.add_scalar('activations/stdev_same_logits', torch.std(logits_same), step)
            add_print_hooks(logits_same, writer, 'activations/same', step)

        return {'logits_unpaired': logits,
                'logits_1': logits_x,
                'logits_2': logits_y,
                'logits_same': logits_same,
                'logits_same_sqz': logits_same.squeeze(1),
                'labels_1': labels_x,
                'labels_2': labels_y,
                'labels_same': labels_same}
            # logits_sup, logits_same_sup.squeeze(
            # 1), labels_same_sup, logits_x, logits_y, logits_same_unsup, unsup_labels