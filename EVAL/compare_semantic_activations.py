import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import simple_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
Use methods such as tNSE 
    to compare if the labelling effect makes sense.

To do that we want to grab activations from the semantic layer
in both the semantic model and the discrete model.
"""    

def grab_activations(model, part, version, model_type):
    """
    Receive a loaded model, run each class at a time.
    2. Compute the output matrix which is in (768,) for each image. 

    3. Compute the average 768 vector for each class, and save this vector.

    input:
    -----
        model: trained model
        part: default val_white, can be either train/val/val_white
        version: version of model weights, for now we use date.
        model_type: semantic or discrete
    """
    # test data
    wordvec_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
    directory = data_directory(part=part)
    wnids, indices, categories = load_classes(num_classes=1000, df='ranked')

    # one class at a time,
    # each class, final result is an average 768 vector
    for i in range(len(wnids)):
        wnid = wnids[i]
        category = categories[i]

        gen, steps = simple_generator(
                        directory=directory,
                        classes=[wnid],
                        batch_size=16,
                        seed=42,
                        shuffle=True,
                        subset=None,
                        validation_split=0,
                        class_mode='sparse',  # TODO, bug, categorical leads to error.
                        target_size=(224, 224),
                        preprocessing_function=preprocess_input,
                        horizontal_flip=False)

        # (N, 768)
        proba = model.predict(gen, steps, verbose=1, workers=3)

        # (768,)
        avg_vec = np.mean(proba, axis=0)
        assert avg_vec.shape == (768,)

        # save avg vec
        np.save(f'_computed_activations/{version}/{model_type}/{category}.npy', avg_vec)


def tsne_best(X):
    """
    To deal with tsne's stochasticity,
    we try multiple restarts and only record the trial associated with
    the smallest loss (kl divergence).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X_pca = PCA(n_components=50).fit_transform(X)
    best_kl_loss = None
    best_Y = None

    for i in range(5):
        model = TSNE(n_components=2, n_iter=10000)
        Y = model.fit_transform(X_pca)
        kl_loss = model.kl_divergence_
        print('iter = [{iter}], kl_loss = [{loss}]'.format(iter=i, loss=kl_loss))

        if best_kl_loss is None:
            best_kl_loss = kl_loss
            best_Y = Y
        else:
            if kl_loss < best_kl_loss:
                best_kl_loss = kl_loss
                best_Y = Y
    print('best_kl_loss = ', best_kl_loss)
    return best_Y

def run_tsne(version, model_type):
    """
    Run tSNE on saved avg vectors to compare difference
    between semantic and discrete models.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 2})


    _, _, categories = load_classes(num_classes=1000, df='ranked')
    embed_mtx = []

    for category in categories:
        avg_vec = np.load(f'_computed_activations/{version}/{model_type}/{category}.npy')
        embed_mtx.append(avg_vec)
    
    X = np.array(embed_mtx)
    print('embed_mtx.shape = ', X.shape)

    ### run tsne and plot ###
    Y = tsne_best(X)
    all_x = Y[:, 0]
    all_y = Y[:, 1]

    fig = plt.figure(dpi=1000)
    ax = fig.add_subplot(1, 1, 1)
    default_s = plt.rcParams['lines.markersize'] ** 2
    ax.scatter(all_x, all_y, alpha=0.2, s=default_s/3)

    for i, txt in enumerate(categories):
        text = ax.annotate(txt, (all_x[i], all_y[i]))
        text.set_alpha(0.4)

    plt.savefig(f'_computed_activations/{version}/tsne-{model_type}.pdf')






def execute():
    ######################
    model_type = 'semantic'
    part = 'val_white'
    version = '1-7-20'
    print('### compare semantic activations ###')
    print(f'model: {model_type}')
    print(f'version: {version}')
    print(f'eval on: {part}')
    print('------------------------------------')
    ######################

    model = ready_model(model_type=model_type, version=version)

    grab_activations(model=model, 
                     part=part, 
                     version=version, 
                     model_type=model_type)

    run_tsne(version, model_type)
