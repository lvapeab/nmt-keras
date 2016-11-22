#!/usr/bin/env bash

if [ $# -lt 2 ]
then
    echo "Usage: $0 corpus_file classes_file [-v]"
    echo "Computes the coberture of the items from classes_file in corpus_file"
     echo " -v: Verbose mode."
    exit 1
fi

corpus=$1
classes=$2
verbose_opt=0
while [ $# -ne 0 ]; do
 case $1 in
      "-v") verbose_opt=1
         ;;
    esac
    shift
done

if [ $verbose_opt -eq 1 ]; then
    sed  -E "s/, /\n/g" ${classes} |grep --color=always -Fwf - ${corpus}
fi
    matches=`sed  -E "s/, /\n/g" ${classes} |grep -Fwf - ${corpus} |wc -l`
    total=`wc -l ${corpus} |awk '{print $1}'`
    echo "Matched ${matches}/${total} (0`echo "scale=4; ${matches} / ${total}"|bc -l`)"



