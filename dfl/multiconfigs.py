from dfl.config import BaseConfig
import dfl.config as config
from dfl.main import main

configs = [
    BaseConfig("P", "GG", "cross_entropy", rl_weight=10),
    BaseConfig("G", "G", "min"),
    BaseConfig("Y", "RY", "Y"),
    BaseConfig("Y", "RY", "Y", p=20.0),
]

for bConfig in configs:
    config.conf.name = "T-norm: " + bConfig.t
    config.conf.reset_to(bConfig)
    main()
