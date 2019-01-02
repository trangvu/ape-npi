#!/usr/bin/env bash

set -e

if [[ $# -lt 4 ]]
then
    echo "wrong number of arguments supplied: $#"
    exit 0
fi

if [ -z ${MOSES} ]
then
    echo "variable MOSES undefined"
    exit 0
fi

config_file=`readlink -f $1`
temp_dir=`readlink -f $2`
filename=`readlink -f $3`
output_filename=$4

cores=`lscpu | grep "^CPU(s)\|Processeur(s)" | sed "s/\(CPU(s):\|Processeur(s).:\)\\s*//"`

if [ -d "${temp_dir}" ]
then
    echo "directory ${temp_dir} already exists"
    exit 0
fi

mkdir -p ${temp_dir}
printf "started: "; date
${MOSES}/scripts/training/filter-model-given-input.pl ${temp_dir}/model ${config_file} ${filename} >/dev/null 2>/dev/null
cat ${filename} | sed "s/|//g" | ${MOSES}/bin/moses -f ${temp_dir}/model/moses.ini -threads ${cores} > ${output_filename} 2>/dev/null
rm -rf ${temp_dir}
printf "finished: "; date
