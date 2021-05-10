import os
import numpy as np
import matplotlib.pyplot as plt
from seaborn import violinplot
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras_custom.generators.generator_wrappers import lang_gen, simple_generator
from EVAL.utils.data_utils import data_directory, load_classes
from EVAL.utils.model_utils import ready_model

"""
# TODO: require integration into the new simclr script.
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


### embedding & distance matrix correlation ###
def embedding_n_distance_matrices(version, lossW, part, lang_model=False, useVGG=False, bert=False, sim_func='L2'):
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

    if sim_func == 'distance':
        disMtx = distance_matrix(X, X)
        print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
        # save based on fname
        np.save(f'RESRC_{part}/_distance_matrices/{fname}.npy', disMtx)
        print('distance matrix saved.')
    elif sim_func == 'cosine_sim':
        disMtx = cosine_similarity(X)
        print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
        # save based on fname
        np.save(f'RESRC_{part}/_cosine_sim_matrices/{fname}.npy', disMtx)
        print('cosine similarity matrix saved.')
    elif sim_func == 'cosine_dist':
        disMtx = cosine_distances(X)
        print(f'fname={fname}, disMtx.shape = ', disMtx.shape)
        # save based on fname
        np.save(f'RESRC_{part}/_cosine_dist_matrices/{fname}.npy', disMtx)
        print('cosine distance matrix saved.')


def execute(compute_semantic_activation=False,
            compute_distance_matrices=True,
            compute_RSA=False,
            finer_compare=False,
            dogVSrest=False,
            dogVSrest2=False,
            ):
    ######################
    part = 'val_white'
    lr = 3e-5
    version = '11-11-20'
    w2_depth = 2
    intersect_layer = 'semantic'
    fname1 = 'bert'
    df = 'primate'
    sim_func = 'cosine_dist'
    lossWs = [0, 0.1, 1, 2, 3, 5, 7, 10]
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
                            bert=False,
                            sim_func=sim_func)
    
    if compute_RSA:
        print('RSA across levels of loss...')
        fname2s = []
        for lossW in lossWs:
            fname2s.append(f'version={version}-lossW={lossW}')
        for fname2 in fname2s:
            RSA(fname1, fname2, mtx_type='cosine_dist', part=part)
    
    if finer_compare:
        finer_distance_compare(lossWs, version, part)
    
    if dogVSrest:
        dog2dog_vs_dog2rest(lossWs, version, df, part)

    if dogVSrest2:
        print('Dog v dog V2...')
        dog2dog_vs_dog2rest_V2(lossWs, version, df, part)




