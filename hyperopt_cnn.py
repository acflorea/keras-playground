from hyperopt import fmin, tpe, hp
from slackclient import SlackClient
from cifar10_cnn_1 import cifar10_cnn_do, slackIt

import sys
import uuid
import os

sys.path.append("./")


def cifar10_cnn(sample):
    conv_layers = sample['conv_layers']
    maps_1 = sample['maps_1']
    maps_2 = sample['maps_2']
    maps_3 = sample['maps_3']
    maps_4 = sample['maps_4']
    maps_5 = sample['maps_5']
    maps_6 = sample['maps_6']

    full_layers = sample['full_layers']
    neurons_1 = sample['neurons_1']
    neurons_2 = sample['neurons_2']
    neurons_3 = sample['neurons_3']
    neurons_4 = sample['neurons_4']

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

    slackIt(sc, 'Start Hyperopt TPE optimisation', slack_channel)

    return cifar10_cnn_do(32, iconv_layers, conv_map.split(','), False, 5, ifull_layers, full_map.split(','),
                          model_name,
                          10, save_dir, False, sc, slack_channel)


space = {

    'conv_layers': hp.choice('conv_layers', [(3), (4), (5), (6)]),

    'maps_1': hp.uniform('maps_1', 100, 1024),
    'maps_2': hp.uniform('maps_2', 100, 1024),
    'maps_3': hp.uniform('maps_3', 100, 1024),
    'maps_4': hp.uniform('maps_4', 100, 1024),
    'maps_5': hp.uniform('maps_5', 100, 1024),
    'maps_6': hp.uniform('maps_6', 100, 1024),

    'full_layers': hp.choice('full_layers', [(1), (2), (3), (4)]),

    'neurons_1': hp.uniform('neurons_1', 1024, 2048),
    'neurons_2': hp.uniform('neurons_2', 1024, 2048),
    'neurons_3': hp.uniform('neurons_3', 1024, 2048),
    'neurons_4': hp.uniform('neurons_4', 1024, 2048),
}

best = fmin(fn=cifar10_cnn,
            space=space,
            algo=tpe.suggest,
            max_evals=1000)
print best
