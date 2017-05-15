#!/usr/bin/env bash

ori_task=europarl
new_task=emea
src_lan=en
trg_lan=fr
split=dev
base_model_path=/media/HDD_2TB/MODELS/online_learning/
model_path=${base_model_path}/${ori_task}_${src_lan}${trg_lan}_GroundHogModel_src_emb_512_bidir_True_enc_LSTM_512_dec_LSTM_512_deepout_linear_trg_emb_512_Adadelta_1.0
model_files=update_300000
dest_dir=Online_learning_experiments/${ori_task}_${new_task}/${src_lan}${trg_lan}/
verbose=1
mkdir -p  $dest_dir

src=/media/HDD_2TB/DATASETS/${new_task}/${src_lan}${trg_lan}/${new_task}_${ori_task}/${split}.${src_lan}
trg=/media/HDD_2TB/DATASETS/${new_task}/${src_lan}${trg_lan}/${new_task}_${ori_task}/${split}.${trg_lan}

for algo in PAS ; do
        for lr in  0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001 ; do
            for c in 1000 100 10 1 0.1 0.001 0.0001 0.00001; do
            echo "`date` Algo: ${algo}. LR: ${lr}. C: ${c}"
            echo -e "\t Storing log in ${dest_dir}/log.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
            hyps=${dest_dir}/hyps.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}
            python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${ori_task}_${src_lan}${trg_lan}.pkl \
                   -src $src  \
                   -trg $trg \
                   -hyp ${hyps} \
                   --models ${model_path}/${model_files} -o -v ${verbose} \
                   --changes LR=${lr} OPTIMIZER=${algo} C=${c}> ${dest_dir}/log.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
            sleep 3
            #echo "`calc_bleu -r ${trg} -t ${hyps}`"
        done
    done
done
echo "Finished"