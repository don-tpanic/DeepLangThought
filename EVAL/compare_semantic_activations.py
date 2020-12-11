import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import lang_gen, simple_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
Use methods such as tNSE or RSA
    to compare if the labelling effect makes sense.

To do that we want to grab activations from the semantic layer
in both the semantic model and the discrete model.
"""    

def grab_activations(model, part, version, lossW):
    """
    Receive a loaded model, run each class at a time.
    2. Compute the output matrix which is in (768,) for each image. 

    3. Compute the average 768 vector for each class, and save this vector.

    input:
    -----
        model: trained model
        part: default val_white, can be either train/val/val_white
        version: version of model weights, for now we use date.
        lossW: weight on the discrete term.
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

        # check if some classes have been computed so can skip
        save_path = f'RESRC_{part}/_computed_activations/{version}/lossW={lossW}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file2save = os.path.join(save_path, f'{category}.npy')
        if not os.path.exists(file2save):
            gen, steps = simple_generator(
                            directory=directory,
                            classes=[wnid],
                            batch_size=128,
                            seed=42,
                            shuffle=True,
                            subset=None,
                            validation_split=0,
                            class_mode='sparse',  # only used for lang due to BERT indexing
                            target_size=(224, 224),
                            preprocessing_function=preprocess_input,
                            horizontal_flip=False)
            # (N, 768)
            proba = model.predict(gen, steps, verbose=1, workers=3)

            # (768,)
            avg_vec = np.mean(proba, axis=0)
            assert avg_vec.shape == (768,)

            # save avg vec
            np.save(file2save, avg_vec)
            print(f'CHECK: saved {file2save}.')


### tsne ###
def tsne_best(X, max_epochs=5):
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

    for i in range(max_epochs):
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

def run_tsne(version, lossW):
    """
    Run tSNE on saved avg vectors to compare difference
    between semantic and discrete models.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 2})


    _, _, categories = load_classes(num_classes=1000, df='ranked')
    embed_mtx = []

    for category in categories:
        avg_vec = np.load(f'_computed_activations/{version}/lossW={lossW}/{category}.npy')
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

    plt.savefig(f'_computed_activations/{version}/lossW={lossW}/tsne.pdf')
### ###


### embedding & distance matrix correlation ###
def embedding_n_distance_matrices(version, lossW, part, lang_model=False, useVGG=False, bert=False):
    """
    Given a model,
    compute a embedding and a distance matrix for targeted activations.
    This is going to save the matrices to make RSA faster.

    outputs:
    --------
        . the saved result is the entire distance matrix (symmetric)
            or the embedding matrix
    """
    # load bert embedding matrix
    if bert:
        embed_mtx = np.load('data_local/imagenet2vec/imagenet2vec_1k.npy')
        fname = 'bert'
    # other than bert, get embedding matrix for vgg or language model
    else:
        _, _, categories = load_classes(num_classes=1000, df='ranked')
        embed_mtx = []

        for category in categories:
            if useVGG:
                avg_vec = np.load(f'RESRC_{part}/_computed_activations/block4_pool=vgg16/{category}.npy')
                fname = 'block4_pool'
            elif lang_model:
                avg_vec = np.load(f'RESRC_{part}/_computed_activations/{version}/lossW={lossW}/{category}.npy')
                fname = f'version={version}-lossW={lossW}'
            embed_mtx.append(avg_vec)

    X = np.array(embed_mtx)
    print(f'fname={fname}, X.shape = ', X.shape)
    np.save(f'RESRC_{part}/_embedding_matrices/{fname}.npy', X)
    print('embedding matrix saved.')


    disMtx = distance_matrix(X, X)
    print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
    # save based on fname
    np.save(f'RESRC_{part}/_distance_matrices/{fname}.npy', disMtx)
    print('distance matrix saved.')


def RSA(fname1, fname2, mtx_type='distance', part='val_white'):
    """
    Supply two models' distance matrices
    or embedding matrices,
    and compute spearmanr between them

    inputs:
    -------
        names for two pre-computed distance matrices.
        mtx_type: either distance or embedding matrices.
    """
    from scipy.stats import spearmanr

    mtx1 = np.load(f'RESRC_{part}/_{mtx_type}_matrices/{fname1}.npy')
    mtx2 = np.load(f'RESRC_{part}/_{mtx_type}_matrices/{fname2}.npy')
    assert mtx1.shape == mtx2.shape

    print(f'**** {fname1} vs {fname2} ****')
    if mtx_type == 'distance':
        uptri1 = mtx1[np.triu_indices(mtx1.shape[0])]
        uptri2 = mtx2[np.triu_indices(mtx2.shape[0])]
        print('uptri spearman', spearmanr(uptri1, uptri2))

    elif mtx_type == 'embedding':
        print('emb spearman', spearmanr(mtx1.flatten(), mtx2.flatten()))
        #print('emb pearson', pearsonr(mtx1.flatten(), mtx2.flatten()))


### ###


### finer compare ###
def finer_distance_compare(lossWs, version, part):
    """
    After showing high correlation figure across
    levels of discrete pressure,

    A closer look is needed into change of distance between 
    a subset of classes to see if discrete pressure leads
    to systematic changes in the semantic space.
    """
    print('version = ', version)
    # 1. overall distance for subset of classes
    ###################
    num_classes = 1000
    df = 'ranked'
    ###################
    wnids, indices, categories = load_classes(num_classes=num_classes, df=df)
    print('len of df = ', len(indices))

    bins = 20
    fig, ax = plt.subplots()
    for i in range(len(lossWs)):

        lossW = lossWs[i]

        distMtx = np.load(f'RESRC_{part}/_distance_matrices/version={version}-lossW={lossW}.npy')      
        subMtx = distMtx[indices, :][:, indices]
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
        print('subMtx.shape = ', subMtx.shape)
        print('subMtx_uptri.shape = ', subMtx_uptri.shape)
        sum_dist = np.sum(subMtx_uptri)
        mean_dist = np.mean(subMtx_uptri)
        std_dist = np.std(subMtx_uptri)
        print(f'lossW={lossW}, sum={sum_dist}, mean={mean_dist}, std={std_dist}')
    
        # 3. error bars.
        ax.errorbar(lossW, mean_dist, yerr=std_dist, capsize=3, fmt='o', label=f'lossW={lossW}')
        ax.set_xlabel('Weight on discrete loss')
        ax.set_ylabel('Relative distance')

    ax.legend()
    if df == 'ranked':
        ax.set_title('Across all 1k classes')
        plt.savefig(f'RESULTS/{part}/{version}-indDistance-{part}.pdf')
        print('plotted.')
    else:
        # # TODO: temp adding lossW=1-semantic=0
        # distMtx = np.load(f'_distance_matrices/version={version}-lossW=1-sup=canidae-semantic=0.npy')      
        # subMtx = distMtx[indices, :][:, indices]
        # subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
        # print('subMtx.shape = ', subMtx.shape)
        # print('subMtx_uptri.shape = ', subMtx_uptri.shape)
        # sum_dist = np.sum(subMtx_uptri)
        # mean_dist = np.mean(subMtx_uptri)
        # std_dist = np.std(subMtx_uptri)
        # print(f'lossW=1, sum={sum_dist}, mean={mean_dist}, std={std_dist}')

        ax.set_title(f'Across subset of {df} classes')
        #plt.savefig(f'RESULTS/{version}-indDistance-{part}.pdf')
        print('plotted.')
    

# TODO: consider integrate back into the above function later.
# TODO: accomodate for other supGroups such as fish etc.
def dog2dog_vs_dog2rest(lossWs, version, df, part):
    """
    The above comparison looks at the change of distance
    across lossWs within a subset of classes, e.g. amoung dogs.

    Here I look at both the above change within a subset, as well 
    as its relationship to the rest of the classes.
    E.g. I hope to see when more pressure on discrete loss, the overall dog cluster
         moves away from the rest.
    """
    #################
    num_classes = 129
    #################
    wnids, indices, categories = load_classes(num_classes=num_classes, df=df)
    bins = 20
    fig, ax = plt.subplots()
    ratios = []    # ratio btw dog2dog and dog2rest
    for i in range(len(lossWs)):

        lossW = lossWs[i]
        # the entire 1k*1k matrix
        distMtx = np.load(f'RESRC_{part}/_distance_matrices/version={version}-lossW={lossW}-sup={df}.npy')
        #print('WARNING: regular model used on superordinates!!')
        # the dogs matrix      
        subMtx = distMtx[indices, :][:, indices]
        # the uptri of dogs matrix
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
        # what we already know about dog vs dog
        mean_dist = np.mean(subMtx_uptri)
        std_dist = np.std(subMtx_uptri)
        # ------------------------------------------------------
        # new stuff: dog vs rest
        nonDog_indices = [i for i in range(1000) if i not in indices]
        # shape = (129, 871)
        dogVSrest_mtx = distMtx[indices, :][:, nonDog_indices]
        dogVSrest_mean_dist = np.mean(dogVSrest_mtx)
        dogVSrest_std_dist = np.std(dogVSrest_mtx)

        ratio = mean_dist / dogVSrest_mean_dist
        ratios.append(ratio)
        print(f'dog2dog = {mean_dist}(std = {std_dist}')
        print(f'dog2rest = {dogVSrest_mean_dist}(std = {dogVSrest_std_dist}')

        if df == 'canidae':
            df = 'dog'
        if lossW == 0.1:
            label1 = f'{df} vs {df}'
            label2 = f'{df} vs the rest'
        else:
            label1 = None
            label2 = None
        #ax.errorbar(lossW, mean_dist, yerr=std_dist, capsize=3, fmt='o', color='g', label=label1)
        #ax.errorbar(lossW, dogVSrest_mean_dist, yerr=dogVSrest_std_dist, capsize=3, fmt='o', color='r', label=label2)

    #ax.plot(lossWs, ratios)
    ax.set_xlabel('Weight on discrete loss')
    ax.set_ylabel('Relative distance')
    #ax.legend()
    plt.grid(True)
    ax.set_title(f'{df} vs {df} & {df} vs the rest')
    plt.savefig(f'RESULTS/{part}/version={version}-{df}-interval.pdf')
    print('plotted.')





def dog2dog_vs_dog2cat(lossWs, version, df_1, df_2, num_classes=1000):
    """
    Compare between supordinates.
    """
    fig, ax = plt.subplots()
    diffs = []  # diff between dog2dog and dog2rest
    ratios = [] # ratio btw dog2dog and dog2rest
    for i in range(len(lossWs)):

        lossW = lossWs[i]
        temp_mean_dist = []  # collects df_1 and df_2's mean dist for a lossW
        for df in [df_1, df_2]:
            wnids, indices, categories = load_classes(num_classes=num_classes, df=df)
            # the entire 1k*1k matrix
            distMtx = np.load(f'_distance_matrices/version={version}-lossW={lossW}-sup={df}.npy')
            # the dogs matrix      
            subMtx = distMtx[indices, :][:, indices]
            # the uptri of dogs matrix
            subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
            # what we already know about dog vs dog
            mean_dist = np.mean(subMtx_uptri)
            std_dist = np.std(subMtx_uptri)

            temp_mean_dist.append(mean_dist)

        ratio = temp_mean_dist[0] / temp_mean_dist[1]
        ratios.append(ratio)

    ax.plot(lossWs, ratios)
    ax.set_xlabel('Weight on discrete loss')
    ax.set_ylabel('Relative distance')
    plt.grid(True)
    ax.set_title(f'{df_1} vs {df_2}')
    plt.savefig(f'RESULTS/{df_1}2{df_2}-distPlot-version={version}-normalised.pdf')

    print('This eval is currently problematic due to unclear comparison....Think more...')


def execute(compute_semantic_activation=False,
            compute_distance_matrices=False,
            compute_RSA=True,
            finer_compare=False,
            dogVSrest=False,
            dogVScat=False,
            ):
    ######################
    part = 'val_white'
    lr = 3e-5
    version = '11-11-20-random'
    w2_depth = 2
    intersect_layer = 'semantic'
    fname1 = 'bert'
    df = None

    lossWs = [0.1, 1, 2, 3, 5, 7, 10]
    for lossW in lossWs:
        if df is not None:
            lossW = f'{lossW}-sup={df}'
        run_name = f'{version}-lr={str(lr)}-lossW={lossW}'

        if compute_semantic_activation:
            model = ready_model(w2_depth=w2_depth, 
                                run_name=run_name, 
                                lossW=lossW, 
                                intersect_layer=intersect_layer)
            grab_activations(model=model, 
                            part=part, 
                            version=version,
                            lossW=lossW)
        if compute_distance_matrices:
            embedding_n_distance_matrices(
                            version, lossW,
                            part, 
                            lang_model=True, 
                            useVGG=False, 
                            bert=False)
    
    if compute_RSA:
        fname2s = []
        for lossW in lossWs:
            fname2s.append(f'version={version}-lossW={lossW}')
        for fname2 in fname2s:
            RSA(fname1, fname2, mtx_type='distance', part=part)
    
    if finer_compare:
        finer_distance_compare(lossWs, version, part)
    
    if dogVSrest:
        dog2dog_vs_dog2rest(lossWs, version, df, part)

    if dogVScat:
        dog2dog_vs_dog2cat(lossWs, version, df_1='bird', df_2='reptile')




