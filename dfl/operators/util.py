import math
import torch

eps = 0.000001
# General sigmoid
def sigmoidal(f, pa, pb0, DEBUG):
    f0 = 0  # f(0, 0)
    f1 = 1  # f(1, 1)
    y1 = math.exp(-pa * (f1 + pb0)) + 1
    y2 = math.exp(-pa * (f0 + pb0)) + 1
    c = y1 / (y2 - y1)

    def _sigmoid(a, b):
        # This is a bug: Doesnt work with f's that have a complete reach. If you use this for implications, it starts
        # acting up as for implications f(0, 0) and f(1, 1) are both 1
        if DEBUG:
            if (a != a).any():
                print("nan a")
            if (b != b).any():
                print("nan b")

        r = c * (y2 * torch.sigmoid(pa * (f(a, b) + pb0)) - 1)
        if DEBUG:
            if (r != r).any():
                print("nan")
                import traceback

                traceback.print_stack()
        return r

    return _sigmoid
