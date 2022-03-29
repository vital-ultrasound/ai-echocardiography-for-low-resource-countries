
import yaml
import os
import torch
import numpy as np
import fcd_utils.classifiers as classifiers
#from PIL import Image

classes = ["Background", "Four chamber"]

verbose = False
net = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = []


def get_classes():
    return classes


def initialize(input_size, model_path, verb: bool = False):
    """
    Load the model and initialize the network
    model_path: path where the config file and the model weights are.

    """
    global device
    global net
    global verbose
    global params

    config_filename = '{}/config_echo_classes.yml'.format(model_path)
    if os.path.exists(config_filename):
        with open(config_filename, 'r') as yml:
            config = yaml.load(yml, Loader=yaml.FullLoader)
    else:
        print('Cannot find config file {}'.format(config_filename), flush=True)
        exit(-1)

    verbose = verb
    # load model for video classification
    print('[FourChDetection_worker.py: initialize] load model {}...'.format(model_path))

    #n_output_classes = 2
    net = classifiers.SimpleVideoClassifier(input_size)
    net.to(device)
    net.eval()

    #checkpoint_f = '{}/best_validation_acc_model.pth'.format(model_path)
    checkpoint_f = '{}/model.pth'.format(model_path)

    if verbose:
        print('[FourChDetection_worker.py::initialize() - Load model {}'.format(checkpoint_f))
    state = torch.load(checkpoint_f)
    net.load_state_dict(state['model_state_dict'])

    if verbose:
        print(net)
    return True


def dowork(frames: np.array, verbose=0):
    with torch.no_grad():
        # pre-process the frames. Crop / resize in C++
        #frames = frames.transpose() # maybe do in cpp?
        #im = Image.fromarray(image_cpp)
        #im.save("/home/ag09/data/VITAL/input.png")
        frames = torch.from_numpy(frames).type(torch.float).to(device).unsqueeze(0).unsqueeze(0)/255.0

        # print(frames.shape)

        try:
            out = net(frames)
            #print(out)
            out_index = torch.argmax(out, dim=1)
            #print(out_index)

        except Exception as ex:
            print('[Python exception caught] FourChDetection_worker::do_work() - {}{}'.format(ex, ex.__traceback__.tb_lineno))
            exit(-1)

    out = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
    #out = out.cpu().numpy()
    #print(out)
    return out
