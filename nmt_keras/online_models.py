import logging
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import *
from keras.losses import *
from keras.regularizers import *

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def setOptimizer(params):
    """
    Sets and compiles a new optimizer for the Translation_Model.
    :param params: Dictionary with optimizer parameters
    :return: Compiled Keras optimizer
    """
    if params.get('VERBOSE', 0) > 0:
        logging.info("Preparing optimizer: %s [LR: %s - LOSS: %s - "
                     "CLIP_C %s - CLIP_V  %s - LR_OPTIMIZER_DECAY %s] and compiling." %
                     (str(params['OPTIMIZER']),
                      str(params.get('LR', 0.01)),
                      str(params.get('LOSS', 'categorical_crossentropy')),
                      str(params.get('CLIP_C', 0.)),
                      str(params.get('CLIP_V', 0.)),
                      str(params.get('LR_OPTIMIZER_DECAY', 0.0))
                      ))

    if params['OPTIMIZER'].lower() == 'sgd':
        optimizer = SGD(lr=params.get('LR', 0.01),
                        momentum=params.get('MOMENTUM', 0.0),
                        decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                        nesterov=params.get('NESTEROV_MOMENTUM', False),
                        clipnorm=params.get('CLIP_C', 10.),
                        clipvalue=params.get('CLIP_V', 0.), )

    elif params['OPTIMIZER'].lower() == 'rsmprop':
        optimizer = RMSprop(lr=params.get('LR', 0.001),
                            rho=params.get('RHO', 0.9),
                            decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                            clipnorm=params.get('CLIP_C', 10.),
                            clipvalue=params.get('CLIP_V', 0.))

    elif params['OPTIMIZER'].lower() == 'adagrad':
        optimizer = Adagrad(lr=params.get('LR', 0.01),
                            decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                            clipnorm=params.get('CLIP_C', 10.),
                            clipvalue=params.get('CLIP_V', 0.))

    elif params['OPTIMIZER'].lower() == 'adadelta':
        optimizer = Adadelta(lr=params.get('LR', 1.0),
                             rho=params.get('RHO', 0.9),
                             decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                             clipnorm=params.get('CLIP_C', 10.),
                             clipvalue=params.get('CLIP_V', 0.))

    elif params['OPTIMIZER'].lower() == 'adam':
        optimizer = Adam(lr=params.get('LR', 0.001),
                         beta_1=params.get('BETA_1', 0.9),
                         beta_2=params.get('BETA_2', 0.999),
                         decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                         clipnorm=params.get('CLIP_C', 10.),
                         clipvalue=params.get('CLIP_V', 0.))

    elif params['OPTIMIZER'].lower() == 'adamax':
        optimizer = Adamax(lr=params.get('LR', 0.002),
                           beta_1=params.get('BETA_1', 0.9),
                           beta_2=params.get('BETA_2', 0.999),
                           decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                           clipnorm=params.get('CLIP_C', 10.),
                           clipvalue=params.get('CLIP_V', 0.))

    elif params['OPTIMIZER'].lower() == 'nadam':
        optimizer = Nadam(lr=params.get('LR', 0.002),
                          beta_1=params.get('BETA_1', 0.9),
                          beta_2=params.get('BETA_2', 0.999),
                          schedule_decay=params.get('LR_OPTIMIZER_DECAY', 0.0),
                          clipnorm=params.get('CLIP_C', 10.),
                          clipvalue=params.get('CLIP_V', 0.))

    elif 'pas' in params['OPTIMIZER'].lower():
        optimizer = eval(params['OPTIMIZER'])(params.get('WEIGHT_SHAPES', None),
                                              lr=params['LR'],
                                              c=params['C'],
                                              clipnorm=params.get('CLIP_C', 10.),
                                              clipvalue=params.get('CLIP_V', 0.))
    else:
        logging.error('\tNot supported optimizer!')

    return optimizer


def build_online_models(models, params):
    trainer_models = []
    if params['USE_CUSTOM_LOSS']:
        logging.info('Using custom loss.')
        # Add additional input layer to models in order to train with custom loss function
        for nmt_model in models:
            nmt_model.setParams(params)
            if params['LOSS'] == 'log_diff':
                h_true = Input(name="h_true", batch_shape=tuple([None, None, None]))
                y_true = Input(name="y_true", batch_shape=tuple([None, None, None]))
                state_below_h = Input(name="state_below_h", batch_shape=tuple([None, None]))
                y_pred = nmt_model.model.outputs[0]
                h_pred = nmt_model.model([nmt_model.model.inputs[0], state_below_h])

                loss_out = Lambda(eval(params['LOSS']),
                                  output_shape=(1,),
                                  name=params['LOSS'],
                                  supports_masking=False)([y_true, y_pred, h_true, h_pred])

                trainer_model = Model(inputs=nmt_model.model.inputs + [state_below_h, y_true, h_true],
                                      outputs=loss_out)
            elif params['LOSS'] == 'log_prob_kl_diff':
                h_true = Input(name="h_true", batch_shape=tuple([None, None, None]))
                y_true = Input(name="y_true", batch_shape=tuple([None, None, None]))
                state_below_h = Input(name="state_below_h", batch_shape=tuple([None, None]))
                y_pred = nmt_model.model.outputs[0]
                h_pred = nmt_model.model([nmt_model.model.inputs[0], state_below_h])
                weight = Input(name="weight", batch_shape=[None, 1])

                mask_y = Input(name="mask_y", batch_shape=tuple([None, None]))
                mask_h = Input(name="mask_h", batch_shape=tuple([None, None]))

                loss_out = Lambda(eval(params['LOSS']),
                                  output_shape=(1,),
                                  name=params['LOSS'],
                                  supports_masking=False)([y_true, y_pred, h_true, h_pred, mask_y, mask_h, weight])

                trainer_model = Model(inputs=nmt_model.model.inputs + [state_below_h, weight] + [y_true, h_true, mask_y, mask_h],
                                      outputs=loss_out)
            elif params['LOSS'] == 'minmax_categorical_crossentropy':
                h_true = Input(name="h_true", batch_shape=tuple([None, None, None]))
                y_true = Input(name="y_true", batch_shape=tuple([None, None, None]))
                state_below_h = Input(name="state_below_h", batch_shape=tuple([None, None]))
                y_pred = nmt_model.model.outputs[0]
                h_pred = nmt_model.model([nmt_model.model.inputs[0], state_below_h])
                weight_y = Input(name="weight_y", batch_shape=[None, 1])
                weight_h = Input(name="weight_h", batch_shape=[None, 1])

                mask_y = Input(name="mask_y", batch_shape=tuple([None, None]))
                mask_h = Input(name="mask_h", batch_shape=tuple([None, None]))

                loss_out = Lambda(eval(params['LOSS']),
                                  output_shape=(1,),
                                  name=params['LOSS'],
                                  supports_masking=False)([y_true, y_pred, h_true, h_pred, mask_y, mask_h, weight_y, weight_h])

                trainer_model = Model(inputs=nmt_model.model.inputs + [state_below_h] + [weight_y, weight_h] + [y_true, h_true, mask_y, mask_h],
                                      outputs=loss_out)
            elif params['LOSS'] == 'weighted_log_diff' or params['LOSS'] == 'pas_weighted_log_diff':
                h_true = Input(name="h_true", batch_shape=tuple([None, None, None]))
                y_true = Input(name="y_true", batch_shape=tuple([None, None, None]))
                state_below_h = Input(name="state_below_h", batch_shape=tuple([None, None]))
                h_pred = nmt_model.model([nmt_model.model.inputs[0], state_below_h])
                y_pred = nmt_model.model.outputs[0]
                weight = Input(name="weight", batch_shape=tuple([None, None]))
                mask_y = Input(name="mask_y", batch_shape=tuple([None, None]))
                mask_h = Input(name="mask_h", batch_shape=tuple([None, None]))
                loss_out = Lambda(eval(params['LOSS']),
                                  output_shape=(1,),
                                  name=params['LOSS'],
                                  supports_masking=False)([y_true, y_pred, h_true, h_pred, mask_y, mask_h, weight])
                trainer_model = Model(inputs=nmt_model.model.inputs + [state_below_h, weight] + [y_true, h_true, mask_y, mask_h],
                                      outputs=loss_out)

            elif isinstance(params['LOSS'], list):
                raise Exception(NotImplementedError, 'WIP!')
                state_below_h1 = Input(name="state_below_h1", batch_shape=tuple([None, None]))
                preds_h1 = nmt_model.model([nmt_model.model.inputs[0], state_below_h1])
                y_true = Input(name="y_true", batch_shape=tuple([None, None, None]))
                y_pred = nmt_model.model.outputs[0]
                inputs = [y_true, y_pred, hyp1, preds_h1, weight1, weight2]
                losses = [Lambda(eval(loss), output_shape=(None,),
                                 name=loss, supports_masking=False)(inputs) for loss in params['LOSS']]

                trainer_model = Model(inputs=nmt_model.model.inputs + [state_below_h1] + [y_true, weight1, weight2],
                                      outputs=loss_out)
            elif params['LOSS'] == 'categorical_crossentropy2':
                y_true = Input(name="y_true", batch_shape=tuple([None, None, None]))
                y_pred = nmt_model.model.outputs[0]
                loss_out = Lambda(eval(params['LOSS']),
                                  output_shape=(1,),
                                  name=params['LOSS'],
                                  supports_masking=False)([y_true, y_pred])
                trainer_model = Model(inputs=nmt_model.model.inputs + [y_true],
                                      outputs=loss_out)
            if 'PAS' in params['OPTIMIZER']:
                # Set custom optimizer
                weights = trainer_model.trainable_weights
                # Weights from Keras 2 are already (topologically) sorted!
                if not weights:
                    logging.warning("You don't have any trainable weight!!")
                    params['WEIGHT_SHAPES'] = [(w.name, K.get_variable_shape(w)) for w in weights]

            if isinstance(params['LOSS'], str):
                params['LOSS'] = {params['LOSS']: lambda y_true, y_pred: y_pred}
            elif isinstance(params['LOSS'], list):
                params['LOSS'] = [{loss_name: lambda y_true, y_pred: y_pred} for loss_name in params['LOSS']]
                if params.get('LOSS_WEIGHTS') is None:
                    logging.warning('Loss weights not given! Using the same weight for each loss')
                    params['LOSS_WEIGHTS'] = [1. / len(params['LOSS']) for _ in params['LOSS']]
                else:
                    assert len(params['LOSS_WEIGHTS']) == len(params['LOSS']), 'You should provide a weight' \
                                                                               'for each loss!'

            optimizer = setOptimizer(params)
            trainer_model.compile(loss=params['LOSS'],
                                  optimizer=optimizer,
                                  loss_weights=params.get('LOSS_WEIGHTS', None),
                                  #  As this is online training, we probably won't need sample_weight
                                  sample_weight_mode=None,  # 'temporal' if params['SAMPLE_WEIGHTS'] else None,
                                  metrics=params.get('KERAS_METRICS', []))
            trainer_models.append(trainer_model)
        return trainer_models
    else:
        for nmt_model in models:
            nmt_model.setParams(params)
            nmt_model.setOptimizer()
        return models
