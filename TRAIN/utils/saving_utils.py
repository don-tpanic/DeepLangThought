import numpy as np 


def save_attention_weights(model, attention_mode, description, importance, run, float):
    """
    For saving attention layer weights from the
    `ICML2020 FilterWise Attention models`

    Notes:
    -----
        The weights directory is now hard coded at block4_pool. 
        When having a different attn layer position, need to make 
        this parameter more flexible.

    inputs:
    ------
        description: an individual category or a group category
    """
    ws = model.get_layer('att_layer_1').get_weights()[0]
    weights_path = 'attention_weights/block4_pool/attention={imp}/[hybrid]_{category}-imp{imp}-run{run}-float{float}.npy'.format(category=description, imp=round(importance,3), run=run, float=float)
    print('saving weights to: ', weights_path)
    np.save(weights_path, ws)
    print('attention weights saved.')


def save_continuous_weights(model, run, factory_depth):
    """
    For saving attention factory weights from the 
    `continuous master model`

    inputs:
    ------
        model: trained continous master model.
        run: specific version.
        factory_depth: number of hidden layers hired in the factory.

    return:
    ------
        saving weights in the corresponding directory.
    """
    for i in range(1, factory_depth+1):
        hid_weights = model.get_layer('hidden_layer_%s' % i).get_weights()
        kernel = hid_weights[0]
        bias = hid_weights[1]

        np.save('attention_weights/continuous_master/partial_weights/adv/hid{i}_kernel_run{run}.npy'.format(i=i, run=run), kernel)
        np.save('attention_weights/continuous_master/partial_weights/adv/hid{i}_bias_run{run}.npy'.format(i=i, run=run), bias)

    attn_weights = model.get_layer('produce_attention_weights').get_weights()
    attn_factory_kernel = attn_weights[0]
    np.save('attention_weights/continuous_master/partial_weights/adv/attn_factory_kernel_run{run}.npy'.format(run=run), attn_factory_kernel)
    print('Partial weights saved.')
