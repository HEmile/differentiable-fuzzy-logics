import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import subprocess
from dfl.dataset import get_loaders
from dfl.model import SameNet
from dfl.real_logic import RealLogic
import dfl.config as config
from dfl.config import conf, test_batch_size


def print_gpu_use():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)


def train(
    model,
    real_logic,
    device,
    sup_loader,
    unsup_loader,
    unsup_enumerator,
    optimizer,
    epoch,
    writer,
    step,
):
    model.train()

    for batch_idx, (sup_data, sup_target) in enumerate(sup_loader):
        try:
            unsup_data, unsup_target = next(unsup_enumerator)
        except:
            unsup_enumerator = iter(unsup_loader)
            unsup_data, unsup_target = next(unsup_enumerator)
        sup_data, sup_target = sup_data.to(device), sup_target.to(device)
        unsup_data, unsup_target = unsup_data.to(device), unsup_target.to(device)

        # Set to none is recommend in Performance Tuning Guide.
        optimizer.zero_grad(set_to_none=True)

        writer_p = writer if step % config.log_interval == 0 else None

        result_sup = model(sup_data, sup_target, None, step, config.conf.log_level)
        sup_loss = torch.nn.CrossEntropyLoss()(
            result_sup["logits_unpaired"], sup_target
        )

        if conf.rl_weight > 0 or conf.same_weight > 0:
            result_unsup = model(
                unsup_data, unsup_target, writer_p, step, config.conf.log_level
            )
        if conf.rl_weight > 0:
            rl_loss = real_logic(result_unsup, writer_p, step)
        else:
            rl_loss = 0
        if conf.same_weight > 0:
            logits_same_sup = result_sup["logits_same_sqz"]
            labels_same_sup = result_sup["labels_same"]
            n_pos_exmps = torch.sum(labels_same_sup).item()
            n_neg_exmps = labels_same_sup.size()[0] - n_pos_exmps
            # pos_weight = 1. * (labels_same_sup.size()[0] - pos_exmps) / pos_exmps

            pos_exmps = logits_same_sup[labels_same_sup]
            neg_exmps = logits_same_sup[~labels_same_sup][
                torch.randint(high=n_neg_exmps, size=(n_pos_exmps,)).long()
            ]
            same_exmps = torch.cat([pos_exmps, neg_exmps])
            same_labels = torch.cat(
                [
                    labels_same_sup.new_ones(n_pos_exmps),
                    labels_same_sup.new_zeros(n_pos_exmps),
                ]
            )

            same_loss = torch.nn.BCEWithLogitsLoss()(same_exmps, same_labels.float())
        else:
            same_loss = 0
        loss = conf.rl_weight * rl_loss + sup_loss + conf.same_weight * same_loss
        loss.backward()
        optimizer.step()

        if step % config.log_interval == 0:
            writer.add_scalar("loss/supervised", sup_loss, step)
            writer.add_scalar("loss/real_logic", rl_loss, step)
            writer.add_scalar("loss/same", same_loss, step)
            writer.add_scalar("loss/total", loss, step)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSupLoss: {:.6f}\tRLLoss: {:.6f}\tSameLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(sup_data),
                    len(sup_loader.dataset),
                    100.0 * batch_idx / len(sup_loader),
                    loss.item(),
                    sup_loss.item(),
                    rl_loss,
                    same_loss,
                )
            )
        step += 1
    return unsup_enumerator, step


# Evaluate the model on the test set
def test(model, real_logic, device, test_loader, writer, step):
    model.eval()

    test_sup_loss = 0.0
    test_rl_loss = 0.0
    test_same_loss = 0.0
    correct = 0.0
    correct_same = 0.0
    correct_same_pos = 0.0
    tot_same = 0.0
    tot_pos_same = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            result = model(data, target, None, step, config.conf.log_level)
            test_rl_loss += real_logic(result, None, step).item()

            sup_logits = result["logits_unpaired"]
            logits_same = result["logits_same_sqz"]
            labels_same = result["labels_same"]
            test_sup_loss += torch.nn.CrossEntropyLoss()(
                sup_logits, target
            )  # sum up batch loss
            test_same_loss += torch.nn.BCEWithLogitsLoss()(
                logits_same, labels_same.float()
            )

            pred = sup_logits.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

            _correct_same_pos = (logits_same[labels_same] > 0.0).sum().item()
            correct_same += _correct_same_pos
            correct_same_pos += _correct_same_pos
            correct_same += (logits_same[~labels_same] < 0.0).sum().item()
            tot_same += logits_same.size()[0]
            tot_pos_same += torch.sum(labels_same.float())

    test_sup_loss = test_sup_loss * test_batch_size / len(test_loader.dataset)
    test_rl_loss = test_rl_loss * test_batch_size / len(test_loader.dataset)
    test_same_loss = test_same_loss * test_batch_size / tot_same
    test_loss = test_sup_loss + test_rl_loss + test_same_loss

    accuracy = 100.0 * correct / len(test_loader.dataset)

    writer.add_scalar("accuracy", correct / len(test_loader.dataset), step)
    writer.add_scalar("loss/supervised", test_sup_loss, step)
    writer.add_scalar("loss/real_logic", test_rl_loss, step)
    writer.add_scalar("loss/total", test_loss, step)
    writer.add_scalar("loss/same", test_same_loss / tot_same, step)
    writer.add_scalar("same/accuracy", correct_same / tot_same, step)
    writer.add_scalar("same/recall", correct_same_pos / tot_pos_same.float(), step)
    writer.add_scalar("same/pos", correct_same_pos, step)

    print(
        "\nTest set: Average loss: {:.4f}, Sup loss: {:.4f} Accuracy: {}/{} ({:.0f}%), Same Accuracy: {:.0f}% Test RL Loss: {:.4f}\n".format(
            test_loss,
            test_sup_loss,
            correct,
            len(test_loader.dataset),
            accuracy,
            100.0 * correct_same / tot_same,
            test_rl_loss,
        )
    )
    return accuracy, test_sup_loss


def main():
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(config.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    sup_loader, unsup_loader, test_loader = get_loaders(config.seed)

    model = SameNet().to(device)
    real_logic = RealLogic(
        config.A_clause, conf.A_quant, conf.T, conf.I, conf.G, conf.log_level
    )

    if config.algorithm == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config.conf.lr, momentum=config.conf.momentum
        )
    elif config.algorithm == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.conf.lr)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    writer_train = SummaryWriter("tb_runs/" + conf.experiment_name + "/train")
    writer_val = SummaryWriter("tb_runs/" + conf.experiment_name + "/val")

    unsup_enumerator = iter(unsup_loader)
    step = 0

    best_loss = 4
    best_accuracy = 0
    for epoch in range(conf.epochs):
        unsup_enumerator, step = train(
            model,
            real_logic,
            device,
            sup_loader,
            unsup_loader,
            unsup_enumerator,
            optimizer,
            epoch,
            writer_train,
            step,
        )

        if epoch % config.test_every_x_epoch == 0:
            accuracy, test_sup_loss = test(
                model, real_logic, device, test_loader, writer_val, step
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print("New best accuracy")
            if test_sup_loss < best_loss:
                best_loss = test_sup_loss
                print("New best loss")


if __name__ == "__main__":
    main()
