from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import ChannelWisePReLU as PReLU
from keras.layers.normalization import BatchNormalization, L2_norm
from keras.regularizers import l2
from keras.layers.core import Dropout, Lambda


def Regularize(layer, params, shared_layers=False, name=''):
    """
    Apply the regularization specified in params to the layer
    :param layer: Layer to regularize
    :param params: Params specifying the regularizations to apply
    :param shared_layers: Boolean indicating if we want to get the used layers for applying to a shared-layers model.
    :return: Regularized layer
    """
    shared_layers_list = []

    if params.get('USE_NOISE') and  params['USE_NOISE']:
        if params.get('NOISE_AMOUNT'):
            shared_layers_list.append(GaussianNoise(params['NOISE_AMOUNT'], name=name + '_gaussian_noise'))

        else:
            shared_layers_list.append(GaussianNoise(0.01))


    if params.get('USE_BATCH_NORMALIZATION') and params['USE_BATCH_NORMALIZATION']:
        if params.get('WEIGHT_DECAY'):
            l2_gamma_reg = l2(params['WEIGHT_DECAY'])
            l2_beta_reg = l2(params['WEIGHT_DECAY'])
        else:
            l2_reg = None

        if params.get('BATCH_NORMALIZATION_MODE'):
            bn_mode = params['BATCH_NORMALIZATION_MODE']
        else:
            bn_mode = 0
        shared_layers_list.append(BatchNormalization(mode=bn_mode,
                                                     gamma_regularizer=l2_gamma_reg,
                                                     beta_regularizer=l2_beta_reg,
                                                     name=name + '_batch_normalization'))

    if params.get('USE_PRELU') and params['USE_PRELU']:
        shared_layers_list.append(PReLU(name=name + '_PReLU'))

    if params.get('USE_DROPOUT') and params['USE_DROPOUT']:
        if params.get('DROPOUT_P'):
            shared_layers_list.append(Dropout(params['DROPOUT_P'], name=name + '_dropout'))
        else:
            shared_layers_list.append(Dropout(0.5, name=name + '_dropout'))

    if params.get('USE_L2') and params['USE_L2']:
        shared_layers_list.append(Lambda(L2_norm, name=name + '_L2_norm'))

    # Apply all the previously built shared layers
    for l in shared_layers_list:
        layer = l(layer)
    result = layer

    # Return result or shared layers too
    if shared_layers:
        return [result, shared_layers_list]
    return result
