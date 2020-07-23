import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '4'
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
        np.save(f'_computed_activations/{version}/lossW={lossW}/{category}.npy', avg_vec)


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
def embedding_n_distance_matrices(version, lossW, lang_model=False, useVGG=False, bert=False):
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
                avg_vec = np.load(f'_computed_activations/block4_pool=vgg16/{category}.npy')
                fname = 'block4_pool'
            elif lang_model:
                avg_vec = np.load(f'_computed_activations/{version}/lossW={lossW}/{category}.npy')
                fname = f'version={version}-lossW={lossW}'
            embed_mtx.append(avg_vec)

    X = np.array(embed_mtx)
    print(f'fname={fname}, X.shape = ', X.shape)
    np.save(f'_embedding_matrices/{fname}.npy', X)
    print('embedding matrix saved.')


    disMtx = distance_matrix(X, X)
    print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
    # save based on fname
    np.save(f'_distance_matrices/{fname}.npy', disMtx)
    print('distance matrix saved.')


def embedding_n_distance_matrices_SUPonly(fname, supGroup, mtx_type='distance'):
    """
    Grab the pre-computed embedding or distance from above,
    extract the sub-matrix based on `supGroup`,
    return and save this sub matrix for further analysis.

    inputs:
    -------
        fname: f'version={version}-lossW={lossW}'
        supGroup: e.g. canidae
    """
    mtx = np.load(f'_{mtx_type}_matrices/{fname}.npy')
    _, indices, _ = load_classes(999, df=supGroup)

    # (1000, 768)
    if mtx_type == 'embedding':
        # (n, 768)
        subMtx = mtx[indices, :]
    # (1000, 1000)
    elif mtx_type == 'distance':
        # (n, n)
        subMtx = mtx[indices, :][:, indices]
        print('TODO: check shape')
        exit()
    
    np.save(f'_{mtx_type}_matrices/{fname}-supGroup={supGroup}.npy')



def RSA(fname1, fname2, mtx_type='distance'):
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

    mtx1 = np.load(f'_{mtx_type}_matrices/{fname1}.npy')
    mtx2 = np.load(f'_{mtx_type}_matrices/{fname2}.npy')
    assert mtx1.shape == mtx2.shape

    if mtx_type == 'distance':
        uptri1 = mtx1[np.triu_indices(mtx1.shape[0])]
        uptri2 = mtx2[np.triu_indices(mtx2.shape[0])]
        print('uptri spearman', spearmanr(uptri1, uptri2))
    elif mtx_type == 'embedding':
        print('emb spearman', spearmanr(mtx1.flatten(), mtx2.flatten()))
        print('emb pearson', pearsonr(mtx1.flatten(), mtx2.flatten()))


### ###


### finer compare ###
def finer_distance_compare(fnames):
    """
    After showing high correlation figure across
    levels of discrete pressure,

    A closer look is needed into change of distance between 
    a subset of classes to see if discrete pressure leads
    to systematic changes in the semantic space.
    """
    # 1. overall distance for subset of classes
    ###################
    num_classes = 1000
    df = 'ranked'
    ###################
    wnids, indices, categories = load_classes(num_classes=num_classes, df=df)

    for f in fnames:
        distMtx = np.load(f'_distance_matrices/{f}.npy')
        
        # TODO: why the slicing works differently?
        subMtx = distMtx[indices, :][:, indices]
        subMtx_uptri = subMtx[np.triu_indices(subMtx.shape[0])]
        print('subMtx.shape = ', subMtx.shape)
        print('subMtx_uptri.shape = ', subMtx_uptri.shape)

        sum_dist = np.sum(subMtx_uptri)
        mean_dist = np.mean(subMtx_uptri)
        std_dist = np.std(subMtx_uptri)
        print(f'{f}, sum={sum_dist}, mean={mean_dist}, std={std_dist}')
    
    # 2. histograms of uptri distance matrices
    bins = 20
    fig, ax = plt.subplots()
    ax.hist(collector[0], label='lossW=10', bins=bins, alpha=0.5)
    ax.hist(collector[1], label='lossW=1', bins=bins, alpha=0.5)
    ax.hist(collector[2], label='lossW=0.1', bins=bins, alpha=0.5)
    ax.set_xlabel('pair wise distance')
    ax.legend()
    plt.savefig('RESULTS/distHists.pdf')

    # 3. distance difference between distance matrices.


        











def execute(compute_semantic_activation=False,
            compute_distance_matrices=False,
            compute_RSA=False,
            finer_compare=True,
            ):
    ######################
    part = 'val_white'
    lr = 3e-5
    lossW = 0
    version = '20-7-20'
    #discrete_frozen = False
    w2_depth = 2
    run_name = f'{version}-lr={str(lr)}-lossW={lossW}'
    intersect_layer = 'semantic'
    # -------------------
    fname1 = 'bert'
    fname2s = [f'version={version}-lossW={lossW}']
    fnames = [f'version={version}-lossW=0', f'version={version}-lossW=0.1', f'version={version}-lossW=1']
    ######################

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
                          lang_model=True, 
                          useVGG=False, 
                          bert=False)
    
    if compute_RSA:
        for fname2 in fname2s:
            RSA(fname1, fname2, mtx_type='embedding')
    
    if finer_compare:
        finer_distance_compare(fnames)




