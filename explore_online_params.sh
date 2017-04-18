#!/usr/bin/env bash

task=xerox
src_lan=en
trg_lan=fr
split=test
base_model_path=/media/HDD_2TB/MODELS/${task}/${src_lan}${trg_lan}/trained_models
model_path=${base_model_path}
model_files=epoch_39
dest_dir=Online_learning_experiments/$task/${src_lan}${trg_lan}/
verbose=1

mkdir -p  $dest_dir





algo=SGD
lr=0.001

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1



algo=Adagrad
lr=0.001

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1



algo=Adagrad
lr=0.0001

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1




algo=Adadelta
lr=0.1

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1




algo=Adadelta
lr=0.01

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1





algo=Adam
lr=0.001

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1





algo=Adam
lr=0.0001

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1




algo=Adam
lr=0.00001

echo " `date` Algo: ${algo}. LR: ${lr}."
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr} 2>&1




algo=PAS
lr=0.01
c=0.01

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3

algo=PAS
lr=0.001
c=0.001

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3









algo=PAS
lr=0.01
c=0.1

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3



algo=PAS
lr=0.01
c=1.

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3









algo=PAS
lr=0.001
c=1

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3









algo=PPAS
lr=0.01
c=0.01

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3









algo=PPAS
lr=0.001
c=0.01

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3









algo=PPAS
lr=0.001
c=0.001

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3




algo=PPAS
lr=0.001
c=1

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3









algo=PPAS
lr=0.0001
c=0.0001

echo " `date` Algo: ${algo}. LR: ${lr}. C: ${c}"
echo -e "\t Storing log in ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c}"
python main.py --config ${model_path}/config.pkl --dataset datasets/Dataset_${task}_${src_lan}${trg_lan}.pkl \
       -src /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${src_lan}  \
       -trg /media/HDD_2TB/DATASETS/${task}/${src_lan}${trg_lan}/${split}.${trg_lan} \
       --hypotheses ${dest_dir}/hyps.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} \
       --models ${model_path}/${model_files} -o -v ${verbose} \
       --changes LR=${lr} OPTIMIZER=${algo} C=${c} > ${dest_dir}/log.${task}_${src_lan}${trg_lan}.${algo}.${lr}.${c} 2>&1
sleep 3





