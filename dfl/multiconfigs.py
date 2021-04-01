from dfl.config import BaseConfig
import dfl.config as config
from dfl.main import main

# Symmetric R-implications
# configs = [
#     BaseConfig("P", "GG", "cross_entropy", rl_weight=10),
#     BaseConfig("G", "G", "min"),
#     BaseConfig("Y", "RY", "Y"),
#     BaseConfig("Y", "RY", "Y", p=20.0),
# ]

# Symmetric S, with adam
configs = [
    BaseConfig("P", "GG", "cross_entropy", rl_weight=10, alg="adam"),
    # BaseConfig("G", "G", "min", alg='adam'),
    BaseConfig("Y", "RY", "Y", alg="adam"),
    BaseConfig("Y", "RY", "Y", p=20.0, alg="adam"),
]

# Symmetric R, with adam
# configs = [
#     BaseConfig("P", "RC", "cross_entropy", rl_weight=10, alg="adam"),
#     # BaseConfig("G", "G", "min", alg='adam'),
#     BaseConfig("Y", "Y", "Y", alg="adam"),
#     BaseConfig("Y", "Y", "Y", p=20.0, alg="adam"),
# ]

for bConfig in configs:
    config.conf.name = "T-norm: " + bConfig.t
    config.conf.reset_to(bConfig)
    main()
