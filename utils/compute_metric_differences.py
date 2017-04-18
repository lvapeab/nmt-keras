# The files specified in the files dict have been obtained with the following awk script:
# tail -n +2 Adam |awk '{print $1" "$6}'| sed 's/,$//' > adam
# where Adam is the output of the evaluate_from_file script (invoked with n > 0)

baseline_name = 'baseline'
multiplier = 100. # Set to 100. if you want percentages
files = [('SGD', 'sgd'),
         ('Adagrad', 'adagrad'),
         ('Adadelta', 'adadelta'),
         ('Adam','adam'),
          ('PA','pa'),
          ('PPA','ppa')
         ]
dest_filename = 'bleu_differences'
algo_scores = {}
for (algo, file_name) in files:
    algo_scores[algo] = map(lambda x: x.split(), open(file_name).read().split('\n')[:-1])

baseline = map(lambda x: x.split(), open(baseline_name).read().split('\n')[:-1])
differences = {}
for algo, scores in algo_scores.iteritems():
    differences[algo] = {}
    for i, (n_sents, score) in enumerate(scores):
        baseline_score = float(baseline[i][1])
        differences[algo][n_sents] = float(score)*multiplier - baseline_score*multiplier

out_f = open(dest_filename, 'w')
to_write = 'Sentences ' + ' '.join([name for (name, filename) in files]) + '\n'
out_f.write(to_write)
for i in sorted([int(b[0]) for b in baseline]):
    to_write = str(i) + ' '
    for algo, filename in files:
        to_write += str(differences[algo][str(i)]) + ' '
    to_write += '\n'
    out_f.write(to_write)
out_f.close()






