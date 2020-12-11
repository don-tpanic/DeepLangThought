import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
if os.environ["CUDA_VISIBLE_DEVICES"] == '2':
    print('WARNING: DO NOT USE GPU2 ON LOVE18!')
    exit()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import simple_generator, sup_gen
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model_for_ind_accuracy_eval

"""
Test trained model on white validation set.
"""

def eval_regular_training_models(part, lossWs, version, lr, w2_depth):
    """
    Use trained model to evaluate top1/5 accuracy at each lossW
    """
    RES_PATH = f'RESULTS/{part}/accuracy'
    for lossW in lossWs:
        run_name = f'{version}-lr={str(lr)}-lossW={lossW}'
        per_lossW_path = os.path.join(RES_PATH, run_name)
        print('per_lossW_path= ', per_lossW_path)
        per_lossW_acc = np.zeros((1000, 2))
        if not os.path.exists(per_lossW_path):
            os.mkdir(per_lossW_path)

        # model
        model = ready_model_for_ind_accuracy_eval(
                            w2_depth=w2_depth, 
                            run_name=run_name, 
                            lossW=lossW)
        
        # test data
        directory = data_directory(part=part)
        wnids, indices, _ = load_classes(num_classes=1000, df='ranked')
        for i in range(len(wnids)):
            index = indices[i]
            wnid = wnids[i]
            print(f'*** currently eval index=[{index}] ***')
            gen, steps = simple_generator(
                                directory=directory,
                                classes=[wnid],
                                batch_size=16,
                                seed=42,
                                shuffle=True,
                                subset=None,
                                validation_split=0,
                                class_mode='sparse',  # need sparse to ensure label corrector has effect.
                                target_size=(224, 224),
                                preprocessing_function=preprocess_input,
                                horizontal_flip=False)

            loss, top1acc, top5acc = model.evaluate_generator(gen, steps, verbose=1, workers=3)

            # BUG: top5 all 0.
            per_lossW_acc[i, :] = [top1acc, top5acc]
        
        np.save(per_lossW_path, per_lossW_acc)
        print('per_lossW_acc saved.')

    
def eval_models_w_superordinates(part, lossWs, version, lr, w2_depth, df):
    """
    Use trained model to evaluate top1/5 accuracy at each lossW

    inputs:
    -------
        supGroup: either one of the superordinates.
    """
    RES_PATH = f'RESULTS/{part}/accuracy'
    for lossW in lossWs:
        run_name = f'{version}-lr={str(lr)}-lossW={lossW}-sup={df}'
        per_lossW_path = os.path.join(RES_PATH, run_name)
        per_lossW_acc = np.zeros((1000, 2))
        if not os.path.exists(per_lossW_path):
            os.mkdir(per_lossW_path)

        # model
        model = ready_model_for_ind_accuracy_eval(
                            w2_depth=w2_depth, 
                            run_name=run_name, 
                            lossW=lossW)
        
        # test data
        directory = data_directory(part=part)
        wnids, indices, _ = load_classes(num_classes=1000, df='ranked')
        _, sup_indices, _ = load_classes(num_classes=1000, df=df)
        for i in range(len(wnids)):
            index = indices[i]
            wnid = wnids[i]
            gen, steps = simple_generator(    # TODO. the true labels need to be swapped such that 151 for all dogs.
                                directory=directory,
                                classes=[wnid],
                                batch_size=16,
                                seed=42,
                                shuffle=True,
                                subset=None,
                                validation_split=0,
                                class_mode='sparse',
                                target_size=(224, 224),
                                preprocessing_function=preprocess_input,
                                horizontal_flip=False)
            
            # TODO: check, this is supposed to swap labels to all `151` if supGroup=canidae
            if index in sup_indices:
                gen.classes = [sup_indices[0]] * len(gen.classes)

            loss, top1acc, top5acc = model.evaluate_generator(gen, steps, verbose=1, workers=3)
            per_lossW_acc[i, :] = [top1acc, top5acc]
        
        np.save(per_lossW_path, per_lossW_acc)
        print('per_lossW_acc saved.')


def print_accuraries(version, lr, lossWs, div='ind', df='reptile'):
    """
    print accuracies based on type,
    if div=='int', we print based on individual classes (regular training),
    if div=='sup', we print sup and non-sup for both regular and sup models:

    inputs:
    -------
        div: 'ind' or 'sup' (div: division of results)
        df: ranked or any other sup
    """
    ACC_PATH = 'RESULTS/val_white/accuracy'
    _, ind_indices, _ = load_classes(num_classes=1000, df='ranked')

    # report two sets of acc based on sup and non-sup
    if div == 'sup':
        _, sup_indices, _ = load_classes(num_classes=1000, df=df)
        nonSup_indices = [i for i in ind_indices if i not in sup_indices]
        print('len(sup_indices) = ', len(sup_indices))
        print('len(nonSup_indices) = ', len(nonSup_indices))

        # baseline acc
        base_acc = np.load(os.path.join(ACC_PATH, 'baseline.npy'))
        base_acc_sup = base_acc[sup_indices]
        base_acc_non = base_acc[nonSup_indices]
        print(f'average sup base acc = [{np.mean(base_acc_sup[:, 0])}], std=[{np.std(base_acc_sup[:, 0])}]')
        print(f'average non base acc = [{np.mean(base_acc_non[:, 0])}], std=[{np.std(base_acc_non[:, 0])}]')

        for lossW in lossWs:
            # other acc
            print(f'*** lossW=[{lossW}] ***')
            acc = np.load(os.path.join(ACC_PATH, f'{version}-lr={str(lr)}-lossW={lossW}-sup={df}.npy'))
            acc_sup = acc[sup_indices]
            acc_non = acc[nonSup_indices]
            print(f'average sup acc = [{np.mean(acc_sup[:, 0])}], std=[{np.std(acc_sup[:, 0])}]')
            print(f'average non acc = [{np.mean(acc_non[:, 0])}], std=[{np.std(acc_non[:, 0])}]')
    
    # report ind class acc
    elif div == 'ind':
        base_acc = np.load(os.path.join(ACC_PATH, 'baseline.npy'))
        print(f'average base acc = [{np.mean(base_acc[:, 0])}], std=[{np.std(base_acc[:, 0])}]')

        for lossW in lossWs:
            # other acc
            print(f'*** lossW=[{lossW}] ***')
            acc = np.load(os.path.join(ACC_PATH, f'{version}-lr={str(lr)}-lossW={lossW}.npy'))
            print(f'average acc = [{np.mean(acc[:, 0])}], std=[{np.std(acc[:, 0])}]')


def plot_regular_models_acc(part, lossWs, version, lr, top_n=1):
    """
    Plot top1/top5 individual class accuracy
    on the val_white as lossW increases. At each
    lossW, we plot a violin of 1000 datapoints.
    """
    RES_PATH = f'RESULTS/{part}/accuracy'
    RES = []
    for lossW in lossWs:
        run_name = f'{version}-lr={str(lr)}-lossW={lossW}'
        per_lossW_path = os.path.join(RES_PATH, run_name)
        # (1000, 2)
        per_lossW_acc = np.load(per_lossW_path)
        if top_n == 1:
            per_lossW_acc = per_lossW_acc[:, 0]
        else:
            per_lossW_acc = per_lossW_acc[:, 1]
        RES.append(per_lossW_acc)
    
    # TODO: iso
    plots = ax.violinplot(dataset=RES, 
                          positions=range(1, len(lossWs)+1), 
                          points=10, widths=0.5, 
                          showmeans=True, 
                          showextrema=False, 
                          showmedians=False)
    colors = ['grey'] * len(RES)
    for j in range(len(plots['bodies'])):
        pc = plots['bodies'][j]
        pc.set_facecolor(colors[j])
    quartile1 = []
    medians = []
    quartile3 = []
    for d in data:
        q1, md, q3 = np.percentile(d, [25,50,75])
        quartile1.append(q1)
        medians.append(md)
        quartile3.append(q3)
    ax.vlines(x_positions, quartile1, quartile3, color='k', linestyle='-', lw=1)
    plt.savefig(RES_PATH+f'/regular_models_acc{top_n}.pdf')


def plot_superordinate_models_acc():
    """
    within and outside superordinate class 
    performance should be plotted separately.

    # TODO: one question is whether for regular models, the corresponding 
    # superoridate performance should also be separately plotted and compared.
    # For now, no.
    """
    pass


def execute():
    ###########################
    part = 'val_white'
    lr = 3e-5
    lossWs = [0.1, 1, 2, 3, 5, 7, 10]
    version = '11-11-20-random'
    w2_depth = 2
    df = 'reptile'
    ###########################

    # regular models:
    #eval_regular_training_models(part, lossWs, version, lr, w2_depth)
    #plot_regular_models_acc(part, lossWs, version, lr, top_n=1)

    # with superordinates:
    #eval_models_w_superordinates(part, lossWs, version, lr, w2_depth, df=df)

    print_accuraries(version, lr, lossWs, div='sup', df=df)



