from hyperopt import fmin, tpe, hp


def grievank(x):
    import math

    sum = 0.0
    prod = 1

    for i in range(0, 6):
        sum = sum + x[i] * x[i] * i / 4000
        prod = prod * math.cos(x[i] / math.sqrt(i + 1))

    w = 1 + sum - prod
    return w


space = {
    0: hp.uniform('x1', -600, 600),
    1: hp.uniform('x2', -600, 600),
    2: hp.uniform('x3', -600, 600),
    3: hp.uniform('x4', -600, 600),
    4: hp.uniform('x5', -600, 600),
    5: hp.uniform('x6', -600, 600),
}

best = fmin(fn=grievank,
            space=space,
            algo=tpe.suggest,
            max_evals= 1000)
print best
