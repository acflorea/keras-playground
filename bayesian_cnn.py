"""Example of how to use this bayesian optimization package."""

import sys
import uuid
import os

from slackclient import SlackClient

from cifar10_cnn_1 import cifar10_cnn_do, slackIt

sys.path.append("./")
from bayes_opt import BayesianOptimization


def cifar10_cnn(conv_layers, maps_1, maps_2, maps_3, maps_4, maps_5, maps_6, full_layers, neurons_1, neurons_2,
                neurons_3, neurons_4):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model_' + str(uuid.uuid4()) + '.h5'

    iconv_layers = int(conv_layers)
    ifull_layers = int(full_layers)

    imaps_1 = int(maps_1) if maps_1 else None
    imaps_2 = int(maps_2) if maps_2 else None
    imaps_3 = int(maps_3) if maps_3 else None
    imaps_4 = int(maps_4) if maps_4 else None
    imaps_5 = int(maps_5) if maps_5 else None
    imaps_6 = int(maps_6) if maps_6 else None

    ineurons_1 = int(neurons_1) if neurons_1 else None
    ineurons_2 = int(neurons_2) if neurons_2 else None
    ineurons_3 = int(neurons_3) if neurons_3 else None
    ineurons_4 = int(neurons_4) if neurons_4 else None

    print "+++++++++++"
    print(iconv_layers, imaps_1, imaps_2, imaps_3, imaps_4, imaps_5, imaps_6, ifull_layers, ineurons_1, ineurons_2,
          ineurons_3, ineurons_4)

    conv_map = str(imaps_1) + "," + str(imaps_2) + "," + str(imaps_3) + "," + str(imaps_4) + "," + str(
        imaps_5) + "," + str(imaps_6)

    full_map = str(ineurons_1) + "," + str(ineurons_2) + "," + str(ineurons_3) + "," + str(ineurons_4)

    slack_token = os.environ["SLACK_API_TOKEN"]
    slack_channel = os.environ["SLACK_CHANNEL"]

    sc = SlackClient(slack_token)

    slackIt(sc, 'Start Bayesian optimisation', slack_channel)

    return cifar10_cnn_do(32, iconv_layers, conv_map.split(','), False, 5, ifull_layers, full_map.split(','),
                          model_name,
                          10, save_dir, False, sc, slack_channel)


# Lets find the maximum of a simple quadratic function of two variables
# We create the bayes_opt object and pass the function to be maximized
# together with the parameters names and their bounds.
bo = BayesianOptimization(cifar10_cnn,
                          {'conv_layers': [3, 6],
                           'maps_1': [100, 1024], 'maps_2': [100, 1024], 'maps_3': [100, 1024], 'maps_4': [100, 1024],
                           'maps_5': [100, 1024], 'maps_6': [100, 1024],
                           'full_layers': [1, 4], 'neurons_1': [1024, 2048], 'neurons_2': [1024, 2048],
                           'neurons_3': [1024, 2048], 'neurons_4': [1024, 2048]})

# One of the things we can do with this object is pass points
# which we want the algorithm to probe. A dictionary with the
# parameters names and a list of values to include in the search
# must be given.
# bo.explore({'x': [-1, 3], 'y': [-2, 2]})

# Additionally, if we have any prior knowledge of the behaviour of
# the target function (even if not totally accurate) we can also
# tell that to the optimizer.
# Here we pass a dictionary with 'target' and parameter names as keys and a
# list of corresponding values
# bo.initialize(
#     {
#         'target': [-1, -1],
#         'x': [1, 1],
#         'y': [0, 2]
#     }
# )

# Once we are satisfied with the initialization conditions
# we let the algorithm do its magic by calling the maximize()
# method.
bo.maximize(init_points=5, n_iter=1000, kappa=2)

# The output values can be accessed with self.res
print(bo.res['max'])

# If we are not satisfied with the current results we can pickup from
# where we left, maybe pass some more exploration points to the algorithm
# change any parameters we may choose, and the let it run again.
# bo.explore({'x': [0.6], 'y': [-0.23]})

# Making changes to the gaussian process can impact the algorithm
# dramatically.
# gp_params = {'kernel': None,
#              'alpha': 1e-5}

# Run it again with different acquisition function
# bo.maximize(n_iter=5, acq='ei', **gp_params)

# Finally, we take a look at the final results.
# print(bo.res['max'])
# print(bo.res['all'])
