import numpy as np
import torch

import fcd_utils.classifiers as classifiers

classes = ["Background", "Four chamber"]

verbose = False
net = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = []


def get_classes():
    return classes


def initialize(input_size, model_path, modelname, verb: bool = False):
    """
    Load the model and initialize the network
    model_path: path where the config file and the model weights are.

    """
    global device
    global net
    global verbose

    verbose = verb
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # n_frames_per_clip = 10
    NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP = 5
    n_classes_ = 2
    # net = classifiers.SimpleVideoClassifier(input_size, n_frames_per_clip, n_classes)
    net = classifiers.basicVGG2D_04layers(in_channels=NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP,
                                          num_classes=n_classes_,
                                          n_frames_per_clip=NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP)
    net.to(device)
    net.eval()
    if verbose:
        print(net)
        print("the model is : " + modelname)

    net_params = torch.load('{}/{}'.format(model_path, modelname))
    net.load_state_dict(net_params)
    net.eval()
    return True

    # \/ TOREVIEW config_echo_classes.yml;state['model_state_dict']
    # global params
    # config_filename = '{}/config_echo_classes.yml'.format(model_path)
    # if os.path.exists(config_filename):
    #     with open(config_filename, 'r') as yml:
    #         config = yaml.load(yml, Loader=yaml.FullLoader)
    # else:
    #     print('Cannot find config file {}'.format(config_filename), flush=True)
    #     exit(-1)
    ## load model for video classification
    # print('[FourChDetection_worker.py: initialize] load model {}...'.format(model_path))

    # #checkpoint_f = '{}/best_validation_acc_model.pth'.format(model_path)
    # checkpoint_f = '{}/model.pth'.format(model_path)
    #
    # if verbose:
    #     print('[FourChDetection_worker.py::initialize() - Load model {}'.format(checkpoint_f))
    # state = torch.load(checkpoint_f)
    # net.load_state_dict(state['model_state_dict'])
    #
    # if verbose:
    #     print(net)
    # return True
    # /\ TOREVIEW config_echo_classes.yml;state['model_state_dict']


def dowork(frames: np.array, verbose=0):
    with torch.no_grad():
        # pre-process the frames. Crop / resize in C++
        # frames = frames.transpose() # maybe do in cpp?
        # im = Image.fromarray(frames)
        # im.save("/home/ag09/data/VITAL/input.png")
        frames = torch.from_numpy(frames).type(torch.float).to(device).unsqueeze(0).unsqueeze(0) / 255.0
        print(f' FourCHDetection_worker:dowork(): frames.size() {frames.shape}')

        try:
            out = net(frames)
            print(f' out {out}')
            out_index = torch.argmax(out, dim=1)
            #print(f'out_index {out_index}')

        except Exception as ex:
            print('[Python exception caught] FourChDetection_worker::do_work() - {}{}'.format(ex,
                                                                                              ex.__traceback__.tb_lineno))
            exit(-1)

    out = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
    # out = out.cpu().numpy()
    # print(out)
    return out
