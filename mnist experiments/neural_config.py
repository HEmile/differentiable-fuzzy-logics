test_every_x_epoch = 1000

imp_train_weight = 1.
dr_weight = 1.0
same_weight = 1.

hidden_n_dr = 100

use_multiclass = True
balance_classes = False
use_logit_inputs = True

EXPERIMENT_NAME = 'split_100/neural_dr'
EXPERIMENT_NAME += ' multiclass' + str(use_multiclass) + 'logit_inputs' + str(use_logit_inputs) + 'hidden_n' + str(hidden_n_dr) + '_drw_' + str(dr_weight) \
                   + '_impw_' + str(imp_train_weight) + '_samew_' + str(same_weight)
print(EXPERIMENT_NAME)

lr = 0.01
momentum = 0.5
log_interval = 100
epochs = 50001