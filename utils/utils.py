def update_parameters(params, updates, restrict=False):
    """
    Updates the parameters from params with the ones specified in updates
    :param params: Parameters dictionary to update
    :param updates: Updater dictionary
    :param restrict: If True, parameters from the original dict are not overwritten.
    :return:
    """
    for new_param_key, new_param_value in updates.iteritems():
        if restrict and params.get(new_param_key) is not None:
            params[new_param_key] = new_param_value
        else:
            params[new_param_key] = new_param_value

    return params
