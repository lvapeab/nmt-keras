#!/bin/bash

spearmint_path=`cd ../../../Spearmint;pwd`
nmt_keras_path=`cd ../../;pwd`
mkdir -p ${nmt_keras_path}/spearmint/db
mkdir -p ${nmt_keras_path}/spearmint/logs

#Launch mongodb if it is not already launched

if [ `ps -wuax |grep mongod |wc -l` -lt 2 ]; then
    mongod --fork --logpath ${nmt_keras_path}/spearmint/db/log --dbpath ${nmt_keras_path}/spearmint/db;
fi


${spearmint_path}/spearmint/cleanup.sh ${nmt_keras_path}/meta-optimizers/spearmint/
nohup python ${spearmint_path}/spearmint/main.py  ${nmt_keras_path}/spearmint --config=${nmt_keras_path}/meta-optimizers/spearmint/config.json > ${nmt_keras_path}/spearmint/logs/out.log 2> ${nmt_keras_path}/spearmint/logs/out.err &
