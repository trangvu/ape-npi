#!/usr/bin/env bash

set -e

if [[ $# -lt 8 ]]
then
    echo "wrong number of arguments supplied: $#"
    exit 0
fi

if [ -z ${MOSES} ] || [ -z ${GIZA} ]
then
    echo "variables MOSES and/or GIZA undefined"
    exit 0
fi

model_dir=`readlink -f $1`
data_dir=`readlink -f $2`
corpus=${data_dir}/$3
dev_corpus=${data_dir}/$4
src_ext=$5
trg_ext=$6
lm_corpus=${data_dir}/$7
lm_order=$8

cores=`lscpu | grep "^CPU(s):" | sed "s/CPU(s):\\s*//"`

${MOSES}/bin/lmplz -o ${lm_order} --discount_fallback < ${lm_corpus}.${trg_ext} > ${lm_corpus}.${trg_ext}.arpa

rm -rf ${model_dir}
mkdir -p ${model_dir}

${MOSES}/scripts/training/train-model.perl -root-dir ${model_dir} \
-corpus ${corpus} -f ${src_ext} -e ${trg_ext} -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:${lm_order}:${lm_corpus}.${trg_ext}.arpa \
-mgiza -external-bin-dir ${GIZA} \
-mgiza-cpus ${cores} -cores ${cores} --parallel

${MOSES}/scripts/training/mert-moses.pl ${dev_corpus}.${src_ext} ${dev_corpus}.${trg_ext} \
${MOSES}/bin/moses ${model_dir}/model/moses.ini --mertdir ${MOSES}/bin/ \
--decoder-flags="-threads ${cores}" &> ${model_dir}/tuning.log --working-dir ${model_dir}/mert-work

mv ${model_dir}/mert-work/moses.ini ${model_dir}/moses.tuned.ini
rm -rf ${model_dir}/mert-work
