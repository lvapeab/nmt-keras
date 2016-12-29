#!/usr/bin/env bash
fast_align=/home/lvapeab/smt/software/fast_align/build/fast_align
utilsdir="./"

usage(){
    echo "Aligns a parallel corpus and creates a mapping (stochastic dictionary) of words."
    echo "Usage: $0 --source <string> --target <string> [--dest <string>] [--aligner <string>]  [--source_lan <string>] [--target_lan <string>] [-v] [--help]"
    echo " --source <string>       : Path to the source corpus."
    echo " --target <string>       : Path to the target corpus."
    echo " --dest <string>         : Path to the destination file (in .pkl format)."
    echo "                                  Default: \"dirname source\"/mapping.\"source_lan\"_\"target_lan\".pkl"
    echo " --aligner <string>      : Aligner to use (options: fast_align (default) or giza)"
    echo " --source_lan <string>   : Source language. If unspecified, taken from the source filename"
    echo " --target_lan <string>   : Source language. If unspecified, taken from the source filename"
    echo " -debug                  : After ending, do not delete temporary files"
    echo " -v                      : Verbose mode."
    echo " --help                  : Display this help and exit."
}

source_file=""
source_file_given=0
target_file=""
target_file_given=0
dest_file=""
dest_file_given=0
aligner="fast_align"
source_lan=""
source_lan_given=0
target_lan=""
target_lan_given=0
verbose=0

while [ $# -ne 0 ]; do
 case $1 in
        "--help") usage
            exit 0
            ;;
        "--source") shift
            if [ $# -ne 0 ]; then
                source_file=$1
                source_file_given=1
            else
                source_file=""
                source_file_given=0
            fi
            ;;
        "--target") shift
            if [ $# -ne 0 ]; then
                target_file=$1
                target_file_given=1
            else
                target_file=""
                target_file_given=0
            fi
            ;;
        "--dest") shift
            if [ $# -ne 0 ]; then
                dest_file=$1
                dest_file_given=1
            else
                dest_file=""
                dest_file_given=0
            fi
            ;;
        "--aligner") shift
            if [ $# -ne 0 ]; then
                aligner=$1
            else
                aligner="fast_align"
            fi
            ;;
        "--source_lan") shift
            if [ $# -ne 0 ]; then
                source_lan=$1
                source_lan_given=1
            else
                source_lan=""
                source_lan_given=0
            fi
            ;;
        "--target_lan") shift
            if [ $# -ne 0 ]; then
                target_lan=$1
                target_lan_given=1
            else
                target_lan=""
                target_lan_given=0
            fi
            ;;
        "-debug") debug="-debug"
            ;;
        "-v") verbose=1
            ;;
    esac
    shift
done


# Verify parameters
if [ ${source_file_given} -eq 0 ]; then
    # invalid parameters
    echo "Error: --source option not given"
    echo "Execute: $0 --help for obtaining help"

    exit 1
fi


if [ ${target_file_given} -eq 0 ]; then
    # invalid parameters
    echo "Error: --target option not given"
    echo "Execute: $0 --help for obtaining help"
    exit 1
fi


if [ ${source_lan_given} -eq 0 ]; then
    source_file_base=$(basename "$source_file")
    source_lan="${source_file_base##*.}"
fi


if [ ${target_lan_given} -eq 0 ]; then
    target_file_base=$(basename "$target_file")
    target_lan="${target_file_base##*.}"
fi

# Verify parameters
if [ ${dest_file_given} -eq 0 ]; then
    dest_dir=$(dirname "$source_file")
    dest_file=${dest_dir}/mapping.${source_lan}_${target_lan}.pkl
fi

dest_dir=$(dirname "$dest_file")

if [ ${verbose} -gt 0 ]; then
    echo -e "Configuration:"
    echo -e "\t source_file: ${source_file}"
    echo -e "\t target_file: ${target_file}"
    echo -e "\t dest_file: ${dest_file}"
    echo -e "\t aligner: ${aligner}"
    echo -e "\t source_lan: ${source_lan}"
    echo -e "\t target_lan: ${target_lan}"
fi

if [ ${verbose} -gt 0 ]; then
    echo "Formatting datasets for ${aligner}..."
fi

python ${utilsdir}/format_corpus_for_aligner.py --source ${source_file} --target ${target_file}\
 --dest ${dest_dir}/${source_lan}_${target_lan} --aligner ${aligner}
if [ "${aligner}" == "fast_align" ]; then
    echo "Aligning with $aligner..."
    ${fast_align} -i ${dest_dir}/${source_lan}_${target_lan} -d -v -o -T 0.1 -I 4 -p ${dest_dir}/${source_lan}_${target_lan}.ttables > ${dest_dir}/${source_lan}_${target_lan}.align
elif [ "${aligner}" == "giza" ]; then
    echo "$aligner aligner STILL not supported!"
    exit 1
    echo "python format_giza.py ${target_file} ${target_file} ${dest_dir}/${source_lan}_${target_lan}"
else
    echo "$aligner aligner not supported!"
    exit 1
fi

if [ ${verbose} -gt 0 ]; then
    echo "Corpus aligned"
    echo "Converting alignments to dictionary"
fi

python ${utilsdir}/ttables_to_dict.py --fname ${dest_dir}/${source_lan}_${target_lan}.ttables \
                                      --dest  ${dest_file}  --verbose ${verbose}

echo "Finished! Alignments stored in: ${dest_file}"

if [ "${debug}" != "-debug" ]; then
    if [ ${verbose} -gt 0 ]; then
        echo "Removing temporal files"
    fi
    rm ${dest_dir}/${source_lan}_${target_lan}.ttables
    rm ${dest_dir}/${source_lan}_${target_lan}
    rm ${dest_dir}/${source_lan}_${target_lan}.align
fi
