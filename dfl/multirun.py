import config
from main import main
import dataset_config

for i in range(3):
    config.seed = i + 7
    config.conf.name = "run" + str(config.seed)
    config.conf.reset_experiment()
    main()
