test_every_x_epoch = 50

imp_train_weight = 1.
dr_weight = 1.
same_weight = 1.

hidden_n_dr = 20

use_multiclass = True
balance_classes = True

EXPERIMENT_NAME = 'split_100/neural_drtest6_balanced'
EXPERIMENT_NAME += ' multiclass' + str(use_multiclass) + 'hidden_n' + str(hidden_n_dr) + '_drw_' + str(dr_weight) + '_impw_' + str(imp_train_weight) + '_samew_' + str(same_weight)
print(EXPERIMENT_NAME)

lr = 0.01
momentum = 0.5
log_interval = 100
epochs = 20001