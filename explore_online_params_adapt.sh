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
verbose=0
mkdir -p  $dest_dir

src=/media/HDD_2TB/MODELS/online_learning/data/${ori_task}_${new_task}/${split}.${src_lan}
trg=/media/HDD_2TB/MODELS/online_learning/data/${ori_task}_${new_task}/${split}.${trg_lan}

for algo in PPAS ; do
        for lr in 1.0 ; do
            for c in 0.001  ; do
                for clipVal in 5. ; do
                    echo "`date` Algo: ${algo}. LR: ${lr}. C: ${c}"
                    echo -e "\t Storing log in ${dest_dir}/log.${split}.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
                    hyps=${dest_dir}/hyps.${split}.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}.${clipVal}
                    python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${ori_task}_${src_lan}${trg_lan}.pkl \
                           -src ${src} \
                           -trg ${trg} \
                           -hyp ${hyps} \
                           --models ${model_path}/${model_files} -o -v ${verbose} \
                           --changes LR=${lr} OPTIMIZER=${algo} C=${c} CLIP_C=${clipVal} > ${dest_dir}/log.${split}.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
                    sleep 3
                    echo "`calc_bleu -r ${trg} -t ${hyps}`"
            done
        done
    done
done
echo "Finished"