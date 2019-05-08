#!/bin/bash


if [ $# -lt 1 ] 
then 
    echo -e "Usage: $(basename $0) <models_path> [-m metric] [-n models_to_leave] "
    echo -e  "Removes all but the n best models, according to the val.coco file. "
    echo -e "Options: "
    echo -e "\t -m metric: Remove models according to this metric. By default, Bleu_4"
    echo -e "\t -n models_to_leave: Number of models to leave. By default, 3."
    exit 1
fi


path=$1
n=3
metric="Bleu_4"

while [ $# -ne 0 ]; do
    case $1 in
	"-m") shift
		 if [ $# -ne 0 ]; then
		     metric=$1
		 else
		     tmpdir="Bleu_4"
		 fi
		 ;;
	"-n") shift
		 if [ $# -ne 0 ]; then
		     n=$1
		 else
		     n=3
		 fi
		 ;;
    esac
    shift
done

if [ ! -d "${path}" ]; then
    echo "The directory ${path} does not exist!"
    exit
fi

if [ ! -f "${path}/val.coco" ]; then
    echo "The file ${path}/val.coco does not exist!"
    exit
fi

tmpdir=$(mktemp -d /tmp/conftmp.XXXXXXXXXXX)

# Get the column to sort:
column_of_interest=$(head -n 1 ${path}/val.coco |awk -v metric=${metric} 'BEGIN{FS=","}{for (i=1; i<=NF; i++) if ($i == metric) print i; }')

# Sort file according to the column
if [ "${metric}" == "TER" ]; then
    extra_sort_options=""
else
    extra_sort_options="-r"
fi
    

sort -n -k 5 -t , "${extra_sort_options}" ${path}/val.coco |head -n ${n} > ${tmpdir}/models_to_save

mkdir ${tmpdir}/model
cp ${path}/config.pkl ${tmpdir}/model/
cp ${path}/val.coco ${tmpdir}/model/

while read p; do
    update_to_save=$(echo "$p" | awk 'BEGIN{FS=","}{print $1}')
    cp ${path}/*_${update_to_save}_* ${tmpdir}/model/
    cp ${path}/*_${update_to_save}.* ${tmpdir}/model/
done < ${tmpdir}/models_to_save

rm ${path}/*
mv ${tmpdir}/model/* ${path}/
rm -rf ${tmpdir}





