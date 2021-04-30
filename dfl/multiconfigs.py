from dfl.config import SameConfig, Sum9Config
import dfl.config as config
from dfl.main import main

configs = {}

# Symmetric R-implications
# configs = [
#     SameConfig("P", "GG", "cross_entropy", rl_weight=10),
#     SameConfig("G", "G", "min"),
#     SameConfig("Y", "RY", "Y"),
#     SameConfig("Y", "RY", "Y", p=1.5),
#     SameConfig("Y", "RY", "Y", p=20.0),
# ]

# # Symmetric R, with adam
# configs = [
#     # SameConfig("P", "GG", "cross_entropy", rl_weight=10, alg="adam"),
#     SameConfig("G", "G", "min", alg="adam"),
#     SameConfig("LK", "LK", "LK", alg="adam"),
#     SameConfig("Np", "F", "Np", alg="adam"),
#     SameConfig("LK", "LK", "mean", alg="adam")
#     # SameConfig("Y", "RY", "Y", alg="adam"),
#     # SameConfig("Y", "RY", "Y", p=20.0, alg="adam"),
# ]

# # Symmetric S, with adam
# configs = [
#     SameConfig("P", "RC", "cross_entropy", rl_weight=10, alg="adam"),
#     SameConfig("G", "KD", "min", alg="adam"),
#     SameConfig("Y", "Y", "Y", alg="adam"),
#     SameConfig("Y", "Y", "Y", alg="adam", p=1.5),
#     SameConfig("Y", "Y", "Y", p=20.0, alg="adam"),
# ]

# # Correct Yager (somehow I ran them unclamped??)
# configs = [
#     SameConfig("Y", "RY", "Y", alg="adam"),
#     SameConfig("Y", "RY", "Y", p=20.0, alg="adam"),
#     SameConfig("Y", "Y", "Y", alg="adam"),
#     SameConfig("Y", "Y", "Y", p=20.0, alg="adam"),
# ]
# # Sum9 basics
# configs = [
#     #     Sum9Config("P", "P", "RC", "cross_entropy", "prob_sum", rl_weight=10.0, alg="adam"),
#     #     Sum9Config("Y", "Y", "Y", "Y", "Y", alg="adam"),
#     #     Sum9Config("Y", "Y", "Y", "Y", "Y", alg="adam", p=0.5),
#     #     Sum9Config("Y", "Y", "Y", "Y", "Y", alg="adam", p=1.5),
#     #     Sum9Config("Y", "Y", "Y", "Y", "Y", alg="adam", p=20.0),
#     #     Sum9Config("LK", "LK", "LK", "LK", "LK", alg="adam"),
#     #     Sum9Config("LK", "LK", "LK", "mean", "LK", alg="adam"),
#     #     Sum9Config("G", "G", "KD", "min", "max", alg="adam"),
#     Sum9Config("Np", "Np", "Np", "Np", "Np", alg="adam"),
# ]
#
# # Best config but with S_G, vary Exists/disjunct
# configs = [
#     Sum9Config("Y", "P", "SP", "cross_entropy", "prob_sum", rl_weight=10.0, alg="adam"),
#     Sum9Config("Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, alg="adam"),
#     Sum9Config(
#         "Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, alg="adam", p=20.0
#     ),
#     Sum9Config("Y", "LK", "SP", "cross_entropy", "LK", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "LK", "SP", "cross_entropy", "LK", rl_weight=10.0, alg="adam"),
#     Sum9Config("Y", "G", "SP", "cross_entropy", "max", rl_weight=10.0, alg="adam"),
# ]

# # cross entropy, SP, vary T, Tco, Exists
# configs = [
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum", rl_weight=10.0, alg="adam"),
#     Sum9Config("LK", "LK", "SP", "cross_entropy", "LK", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "LK", "SP", "cross_entropy", "LK", rl_weight=10.0, alg="adam"),
#     Sum9Config("G", "G", "SP", "cross_entropy", "max", rl_weight=10.0, alg="adam"),
#     Sum9Config("Np", "Np", "SP", "cross_entropy", "Np", rl_weight=10.0, alg="adam"),
# ]

# # cross entropy, SP, vary T, Tco, Exists, without adam
# configs = [
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum", lr=0.001, rl_weight=10.0),
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum", lr=0.001),
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum"),
#     Sum9Config(
#         "P", "P", "SP", "cross_entropy", "prob_sum", momentum=0.9, rl_weight=10.0
#     ),
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum", momentum=0.9),
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum", alg="adam"),
#     Sum9Config("P", "P", "SP", "cross_entropy", "prob_sum", lr=0.0001, alg="adam"),
#     Sum9Config(
#         "P",
#         "P",
#         "SP",
#         "cross_entropy",
#         "prob_sum",
#         lr=0.0001,
#         alg="adam",
#         rl_weight=10.0,
#     ),
# ]

# Best config but with S_G, vary Exists
# configs = [
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "max", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "hmean", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "gmean", rl_weight=10.0, alg="adam"),
#     # Sum9Config(
#     #     "Y", "G", "SP", "cross_entropy", "gmean", rl_weight=10.0, p=20
#     # ),
#     Sum9Config("Y", "G", "SP", "cross_entropy", "gmean", rl_weight=10.0, p=1.5),
#     # Sum9Config(
#     #     "Y", "G", "SP", "cross_entropy", "gmean", rl_weight=10.0, alg="adam", p=0.5
#     # ),
#     # Sum9Config("Y", "G", "SP", "mean", "log_prob_sum", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "LK", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "Np", "SP", "cross_entropy", "Np", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "Np", rl_weight=10.0, alg="adam"),
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "Y", rl_weight=10.0, alg="adam", p=0.5),
#     # Sum9Config("Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, alg="adam", p=0.5),
#     # Sum9Config("Y", "G", "SP", "cross_entropy", "Y", rl_weight=10.0, alg="adam", p=2.0),
# ]

# Implication experiments on same problem
configs["implications"] = [
    SameConfig("Y", "KD", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "LK", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "RC", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "Np", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=2.0),  # RERUN, WRONG TP
    SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=1.5),
    SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=20),  # RERUN, WRONG TP
    SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=0.5),  # NAN! RERUN!
    SameConfig("Y", "G", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "GG", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=2.0),
    SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=1.5),
    SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=20),
    SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=0.5),
]

# Best config but with S_G, vary Exists/disjunct
configs["exists"] = [
    Sum9Config("Y", "Y", "SP", "cross_entropy", "max", rl_weight=10.0),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "hmean", rl_weight=10.0),
    # This one is computed in t-norm.
    # Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, alg="adam"),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, ep=20),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, ep=1.5),
    # Sum9Config(
    #     "Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, alg="adam", ep=0.5
    # ),
    Sum9Config("Y", "Y", "SP", "mean", "log_prob_sum", rl_weight=10.0),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "LK", rl_weight=10.0),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "Np", rl_weight=10.0),
    # Sum9Config(
    #     "Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, alg="adam", ep=0.5
    # ),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, ep=2.0),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, ep=1.5),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "Y", rl_weight=10.0, ep=20),
]

configs["t-norm"] = [
    Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, tp=1.5),
    Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, tp=20),
    Sum9Config("P", "P", "SP", "cross_entropy", "gmean", rl_weight=10.0),
    Sum9Config("LK", "LK", "SP", "cross_entropy", "gmean", rl_weight=10.0),
    Sum9Config("G", "G", "SP", "cross_entropy", "gmean", rl_weight=10.0),
    Sum9Config("Np", "Np", "SP", "cross_entropy", "gmean", rl_weight=10.0),
]

configs["forall"] = [
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0),
    SameConfig("Y", "SP", "min", rl_weight=10.0),
    SameConfig("Y", "SP", "LK", rl_weight=10.0),
    SameConfig("Y", "SP", "RMSE", rl_weight=10.0),
    SameConfig("Y", "SP", "GME", rl_weight=10.0, ap=1.5),
    SameConfig("Y", "SP", "GME", rl_weight=10.0, ap=20),
    SameConfig("Y", "SP", "Y", rl_weight=10.0),
    SameConfig("Y", "SP", "Y", rl_weight=10.0, ap=1.5),
    SameConfig("Y", "SP", "Y", rl_weight=10.0, ap=20),
    SameConfig("Y", "SP", "Np", rl_weight=10.0),
]

configs["random"] = [
    # SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=1.5),  # NAN
    # SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=2.0),
    # SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=20),
    # SameConfig("Y", "RY", "cross_entropy", rl_weight=10.0, ip=0.5),
    # SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=2.0),  # RERUN, WRONG TP
    # SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=20),  # RERUN, WRONG TP
    # SameConfig("Y", "Y", "cross_entropy", rl_weight=10.0, ip=0.5),  # NAN! RERUN!
    # Sum9Config("Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, ep=2)
    # SameConfig("Y", "Np", "cross_entropy", rl_weight=10.0),
    # Sum9Config("Y", "Y", "Y", "Y", "Y", alg="adam", ep=1.5, tp=1.5),
    # SameConfig("Y", "Y", "Y", alg="adam", tp=1.5, ap=1.5, ip=1.5),
    # SameConfig("Y", "RY", "Y", alg="adam", tp=1.5, ap=1.5, ip=1.5),
    # Sum9Config(
    # "Y", "Y", "SP", "cross_entropy", "gmean", rl_weight=10.0, ep=1.5, alg="sgd"
    # ),
    SameConfig("Y", "SP", "Y", rl_weight=10.0, ap=1.5),
]

configs["s_experiments"] = [
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=1.0),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=2.5),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=5),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=7.5),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=12.5),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=15.0),
    SameConfig("Y", "SP", "cross_entropy", rl_weight=10.0, s=20.0),
]

configs["LK_rerun"] = [
    SameConfig("LK", "LK", "LK"),
    Sum9Config("LK", "LK", "LK", "LK", "LK"),
]

for bConfig in configs[config.conf.multiconfig]:
    config.conf.name = "T-norm: " + bConfig.t
    if isinstance(bConfig, SameConfig):
        config.conf.reset_to(bConfig)
    elif isinstance(bConfig, config.Sum9Config):
        config.conf.reset_to_sum9(bConfig)
    main()
