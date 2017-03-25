from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import ChannelWisePReLU as PReLU
from keras.layers.normalization import BatchNormalization, L2_norm
from keras.regularizers import l2
from keras.layers.core import Dropout, Lambda


def Regularize(layer, params, shared_layers=False, name=''):
    """
    Apply the regularization specified in parameters to the layer
    :param layer: Layer to regularize
    :param params: Params specifying the regularizations to apply
    :param shared_layers: Boolean indicating if we want to get the used layers for applying to a shared-layers model.
    :param name: Name prepended to regularizer layer
    :return: Regularized layer
    """
    shared_layers_list = []

    if params.get('USE_NOISE', False):
        shared_layers_list.append(GaussianNoise(params.get('NOISE_AMOUNT', 0.01), name=name + '_gaussian_noise'))

    if params.get('USE_BATCH_NORMALIZATION', False):
        if params.get('WEIGHT_DECAY'):
            l2_gamma_reg = l2(params['WEIGHT_DECAY'])
            l2_beta_reg = l2(params['WEIGHT_DECAY'])
        else:
            l2_gamma_reg = None
            l2_beta_reg = None

        bn_mode = params.get('BATCH_NORMALIZATION_MODE', 0)
        shared_layers_list.append(BatchNormalization(mode=bn_mode,
                                                     gamma_regularizer=l2_gamma_reg,
                                                     beta_regularizer=l2_beta_reg,
                                                     name=name + '_batch_normalization'))

    if params.get('USE_PRELU', False):
        shared_layers_list.append(PReLU(name=name + '_PReLU'))

    if params.get('USE_DROPOUT', False) :
        shared_layers_list.append(Dropout(params.get('DROPOUT_P', 0.5), name=name + '_dropout'))

    if params.get('USE_L2', False):
        shared_layers_list.append(Lambda(L2_norm, name=name + '_L2_norm'))

    # Apply all the previously built shared layers
    for l in shared_layers_list:
        layer = l(layer)

    result = layer

    # Return result or shared layers too
    if shared_layers:
        return result, shared_layers_list
    return result
