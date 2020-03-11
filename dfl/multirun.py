import dfl.config as config
from dfl.main import main

for i in range(3):
    config.seed = i + 7
    config.conf.name = "run" + str(config.seed)
    config.conf.reset_experiment()
    main()
