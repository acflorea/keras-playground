import sys
import time

import os

import uuid

import optunity
import optunity.metrics
from optunity import functions as fun
from optunity import search_spaces, api

from cifar10_cnn_1 import cifar10_cnn_do, slackIt

import numpy as np
import tensorflow as tf
import random as rn

from slackclient import SlackClient

def main(args):
    start_time = time.time()

    slack_token = os.environ["SLACK_API_TOKEN"]
    sc = SlackClient(slack_token)

    slackIt(sc, "optunity start :rocket:")

    # The meaning of life should be fixed
    np.random.seed(42)
    rn.seed(42)
    tf.set_random_seed(42)

    num_evals = int(args[1])
    solver_name = args[2]

    print("Input: " + str(args))

    maps = [8, 512]
    neurons = [5, 2048]

    search_space = {
        'conv_layers': {'3': {'maps_1': maps, 'maps_2': maps, 'maps_3': maps},
                        '4': {'maps_1': maps, 'maps_2': maps, 'maps_3': maps, 'maps_4': maps},
                        '5': {'maps_1': maps, 'maps_2': maps, 'maps_3': maps, 'maps_4': maps, 'maps_5': maps},
                        '6': {'maps_1': maps, 'maps_2': maps, 'maps_3': maps, 'maps_4': maps, 'maps_5': maps,
                              'maps_6': maps}
                        },
        'full_layers': {'1': {'neurons_1': neurons},
                        '2': {'neurons_1': neurons, 'neurons_2': neurons},
                        '3': {'neurons_1': neurons, 'neurons_2': neurons, 'neurons_3': neurons},
                        '4': {'neurons_1': neurons, 'neurons_2': neurons, 'neurons_3': neurons, 'neurons_4': neurons}
                        }
    }

    f = cifar10_cnn

    tree = search_spaces.SearchTree(search_space)
    box = tree.to_box()

    # we need to position the call log here
    # because the function signature used later on is internal logic
    f = fun.logged(f)

    # wrap the decoder and constraints for the internal search space representation
    f = tree.wrap_decoder(f)
    f = api._wrap_hard_box_constraints(f, box, -sys.float_info.max)

    # build solver
    suggestion = api.suggest_solver(num_evals, solver_name, **box)
    solver = api.make_solver(**suggestion)

    solution, details = api.optimize(solver, f, maximize=True, max_evals=num_evals, decoder=tree.decode)

    # solution, details = api.optimize(solver, f, maximize=True, max_evals=num_evals,
    #                                  pmap=optunity.pmap, decoder=tree.decode)

    print("Optimal parameters: " + str(solution))
    print("Optimal value: " + str(details.optimum))

    print("--- %s seconds ---" % (time.time() - start_time))


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
    sc = SlackClient(slack_token)

    return  cifar10_cnn_do(32, iconv_layers, conv_map.split(','), False, 5, ifull_layers, full_map.split(','), model_name,
                   10, save_dir, False, sc)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
