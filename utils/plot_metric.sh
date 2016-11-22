#!/bin/bash                                                                                                                                                                     

# Read and plot several logs from cococaption                                                                                                                                   

if [ $# -lt 1 ];
then
    echo "Usage $0 [train.log] [val.log] [test.log]"
fi

metric_pos="3"
metric_name="Bleu_4"
out_name="./${metric_name}_plot"
 tail -n +2 $1 | awk 'BEGIN{FS=","}{print 1}'>/tmp/epochs;

i=1
for result in "$@"; do
    basename=$(basename $result)
    tail -n +2 $result | awk -v pos=${metric_pos} 'BEGIN{FS=","}{print $pos}'>/tmp/${basename};
    names[$i]="${basename%.*}"
    i=$(( i + 1 ))
basenames=${basenames}" /tmp/`basename $result`"
done
echo "Epoch ${names[*]}" > /tmp/scores

paste -d " " /tmp/epochs $basenames  >> /tmp/scores

echo "set encoding iso_8859_1

set style data lines
set key font ',20'   height 2
set xtics font ',18' 
set ytics font ',18' 
set xlabel font ',20'  '# Epoch' 
set ylabel font ',20' '${metric_name}';

set title ''
set terminal pdf enhanced
set termoption dash
set output '${out_name}.pdf'
set key left

set yrange[0:1]
set ytics (0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)

set bmargin 4
plot for [col=2:$(( $# + 1 ))] '/tmp/scores' using 0:col with lines lt col lw 5 title columnheader " | gnuplot
 









