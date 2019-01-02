#!/usr/bin/env bash

set -e

function sort_config {
    tmp=`mktemp`
    grep -Pv "^$|model_dir|data_dir|label|description" $1 | sed "s/^\\s\+-\?\\s*/    /" | grep -Pv "^\\s*#" > ${tmp}
    output=`mktemp`
    grep -Pv "encoders|decoders|reverse_mapping|^[\s]" ${tmp} | sort > ${output}
    echo "decoders:" >> ${output}
    sed -n -e "/encoders/,/^[^ ]/p" ${tmp} | grep "^\s\+" | sort >> ${output}
    echo "encoders:" >> ${output}
    sed -n -e "/decoders/,/^[^ ]/p" ${tmp} | grep "^\s\+" | sort >> ${output}
    rm -f ${tmp}
    echo ${output}
}

filename1=`sort_config $1`
filename2=`sort_config $2`

sdiff -dbBWZs ${filename1} ${filename2}
rm -f ${filename1} ${filename2}

