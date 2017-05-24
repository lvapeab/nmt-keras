#!/bin/bash

spearmint_path=${SOFTWARE_PREFIX}/Spearmint
nmt_keras_path=${SOFTWARE_PREFIX}/nmt-keras
dest_dir=${nmt_keras_path}/meta-optimizers/spearmint
mkdir -p ${dest_dir}/db
mkdir -p ${dest_dir}/logs

#Launch mongodb if it is not already launched
if [ $(ps -wuax |grep mongod |wc -l) -lt 2 ]; then
    mongod --fork --logpath ${dest_dir}/db/log --dbpath ${dest_dir}/db;
fi


${spearmint_path}/spearmint/cleanup.sh ${dest_dir}

cd ${nmt_keras_path}; nohup python ${spearmint_path}/spearmint/main.py ${dest_dir} --config=${nmt_keras_path}/meta-optimizers/spearmint/config.json >> ${dest_dir}/logs/out.log 2> ${dest_dir}/logs/out.err &
echo "Main Spearmint process PID:" $! >> ${dest_dir}/logs/out.log