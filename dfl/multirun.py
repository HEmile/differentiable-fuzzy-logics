import dfl.config as config
from dfl.main import main

for i in range(10):
    # Fix seed instead of setting different one (fixed split of data)
    config.seed = i + 7
    config.conf.name = "run" + str(config.seed)
    config.conf.reset_experiment()
    main()
