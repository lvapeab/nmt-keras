#!/usr/bin/env bash



ori_task=europarl
new_task=emea
src_lan=en
trg_lan=fr
split=test
base_model_path=/media/HDD_2TB/MODELS/${ori_task}/trained_models
model_path=${base_model_path}/${ori_task}_${src_lan}${trg_lan}_GroundHogModel_src_emb_512_bidir_True_enc_LSTM_512_dec_LSTM_512_deepout_linear_trg_emb_512_Adadelta_1.0joint_bpe
model_files=update_300000
dest_dir=Online_learning_experiments/${ori_task}_${new_task}/${src_lan}${trg_lan}/
verbose=1

mkdir -p  $dest_dir

src=/media/HDD_2TB/DATASETS/${new_task}/${src_lan}${trg_lan}/${new_task}_${ori_task}/${split}.${src_lan}
trg=/media/HDD_2TB/DATASETS/${new_task}/${src_lan}${trg_lan}/${new_task}_${ori_task}/${split}.${trg_lan}

algo=Adadelta
lr=0.01

echo "`date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${ori_task}_${src_lan}${trg_lan}.pkl \
       -src $src  \
       -trg $trg \
       -hyp ${dest_dir}/hyps.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${ori_task}_${new_task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1
