import logging
import subprocess
import os
import sys
#sys.path.append("../../") # Adds higher directory to python modules path.
sys.path.insert(1, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../"))

print sys.path

from config import load_parameters
from main import check_params, train_model
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
metric_name = 'Bleu_4'
maximize = True  # Select whether we want to maximize the metric or minimize it
d = dict(os.environ.copy())
d['LC_NUMERIC'] = 'en_US.utf-8'
def invoke_model(parameters):
    """
    Loads a model, trains it and evaluates it.
    :param parameters: Model parameters
    :return: Metric to minimize value.
    """

    model_params = load_parameters()
    model_name = model_params["MODEL_TYPE"]
    for parameter in parameters.keys():
        model_params[parameter] = parameters[parameter][0]
        logger.debug("Assigning to %s the value %s" % (str(parameter), parameters[parameter][0]))
        model_name += '_' + str(parameter) + '_' + str(parameters[parameter][0])
    model_params["MODEL_NAME"] = model_name
    # models and evaluation results will be stored here
    model_params["STORE_PATH"] = 'trained_models/' + model_params["MODEL_NAME"] + '/'
    check_params(model_params)
    assert model_params['MODE'] == 'training', 'You can only launch Spearmint when training!'
    logging.info('Running training.')
    train_model(model_params)

    results_path = model_params['STORE_PATH'] + '/' + model_params['EVAL_ON_SETS'][0] + '.' + model_params['METRICS'][0]

    # Recover the highest metric score
    metric_pos_cmd = "head -n 1 " + results_path + \
                     " |awk -v metric=" + metric_name + \
                     " 'BEGIN{FS=\",\"}" \
                     "{for (i=1; i<=NF; i++) if ($i == metric) print i;}'"
    metric_pos = subprocess.Popen(metric_pos_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True).communicate()[0][:-1]
    cmd = "tail -n +2 " + results_path + \
          " |awk -v m_pos=" + str(metric_pos) + \
          " 'BEGIN{FS=\",\"}{print $m_pos}'|sort -gr|head -n 1"
    ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, env=d)
    metric_value = float(ps.communicate()[0])
    print "Best %s: %f" % (metric_name, metric_value)

    return 1. - metric_value if maximize else metric_value  # Spearmint minimizes a function


def main(job_id, params):
    """
    Launches the spearmint job
    :param job_id: Job identifier.
    :param params: Model parameters.
    :return: Metric to minimize value.
    """
    print params
    return invoke_model(params)

if __name__ == "__main__":
    # Testing function
    params = {'SOURCE_TEXT_EMBEDDING_SIZE': [1],
              'ENCODER_HIDDEN_SIZE': [2],
              'TARGET_TEXT_EMBEDDING_SIZE': [1],
              'DECODER_HIDDEN_SIZE': [2],
              'MAX_EPOCH': [2],
              'START_EVAL_ON_EPOCH': [1]}
    main(1, params)
