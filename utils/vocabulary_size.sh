#!/bin/bash


if [ $# -lt 1 ] 
then 
    echo "Usage: $0 text_file"
    echo "Computes the vocabulary size of text_file"
    exit 1
fi


for file in  $* ;do
  vocab=$(tr " " '\n' <${file}| sort -u |wc -l)
  echo "$file: $vocab"
done
