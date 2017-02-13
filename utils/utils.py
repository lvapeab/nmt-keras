
def update_parameters(params, updates):
    """
    Updates the parameters from params with the ones specified in updates
    :param params: Parameters dictionary to update
    :param updates: Updater dictionary
    :return:
    """
    for new_param_key, new_param_value in updates.iteritems():
        params[new_param_key] = new_param_value

    return params