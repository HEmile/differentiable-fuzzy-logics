import math
import matplotlib.pyplot as plt


def c(i, b):
    # Sample b times with replacement, with i different options.
    # Then this function returns the amount of configurations where there is at least one of each option sampled
    return i ** b - sum(map(lambda j: math.comb(i, j) * c(j, b), range(1, i)))


batch_size = 100
print(1 - (c(10, batch_size) / 10 ** batch_size))

# This is a log-linear function in batch_size, see this image:
fig, ax = plt.subplots()
ax.plot(
    list(range(1, 300)),
    list(map(lambda a: math.log(1 - (c(10, a) / 10 ** a)), range(1, 300))),
)  # Plot some data on the axes.
plt.show()
