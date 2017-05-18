#!/usr/bin/env bash

task=emea
src_lan=en
trg_lan=fr
split=dev
base_model_path=/media/HDD_2TB/MODELS/online_learning/
model_path=${base_model_path}/${task}_${src_lan}${trg_lan}_GroundHogModel_src_emb_512_bidir_True_enc_LSTM_512_dec_LSTM_512_deepout_linear_trg_emb_512_Adadelta_1.0
model_files=update_130000 
dest_dir=Online_learning_experiments/$task/${src_lan}${trg_lan}/
verbose=0

mkdir -p  $dest_dir
for algo in PAS  ; do
	#for lr in 0.1 ; do #0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001 ; do
	#    for c in 10 1 0.1 0.001 0.0001 0.00001; do

	for lr in 0.1; do #0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001 ; do
        for c in 0.1 ; do
	    echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
            echo -e "\t Storing log in ${dest_dir}/log.${split}.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
            python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
                   -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
                   -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
                   --hypotheses ${dest_dir}/hyps.OLD3.${split}.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
                   --models ${model_path}/${model_files} -o -v ${verbose} \
                   --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${split}.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
            sleep 3
        done
    done
done
echo "Finished!"
