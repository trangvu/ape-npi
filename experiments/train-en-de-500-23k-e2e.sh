#!/bin/bash
#SBATCH --job-name=500-23k-small
#SBATCH --account=da33
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60000
#SBATCH --gres=gpu:1
#SBATCH --partition=m3h
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vuth0001@student.monash.edu
#SBATCH --output=500-23k-small-%j.out
#SBATCH --error=500-23k-small-%j.err

LM_SRC_PATH="/home/xvuthith/da33/trang/rnn-lm"
LM_DATA_DIR="/home/xvuthith/da33/trang/dataset/IT-More-Data-APE/data"
LM_VOCAB_PATH="/home/xvuthith/da33/trang/dataset/IT-More-Data-APE/data/vocab.de"

APE_SRC_PATH="/home/xvuthith/da33/trang/ape"
APE_CONF_PATH="/home/xvuthith/da33/trang/jobs/experiment/mt+ag+lm/config"
APE_DATA_DIR="/home/xvuthith/da33/trang/dataset/en_de/APE17_LIG"

module load tensorflow/1.4.0-python3.6-gcc5
module load cuda/8.0
module load python/3.6.2
module load cudnn/5.1
module load java/1.7.0_67
source /home/xvuthith/da33/trang/jobs/envname/bin/activate

DATE=`date '+%Y%m%d-%H%M%S'`
WK_DIR="/home/xvuthith/da33/trang/jobs/experiment/500-23k-end2end-"$DATE
mkdir -p $WK_DIR

echo "STEP 1: Train MT + AG"
MT_AG_CONFIG_FILE="config-MT-AG-en-de.yaml"
cp $APE_CONF_PATH/origin-emb-syn.yaml $WK_DIR/$MT_AG_CONFIG_FILE
sed -i "/^model_dir:/c\model_dir: $WK_DIR/mt-ag/model" $WK_DIR/$MT_AG_CONFIG_FILE
sed -i "/^log_dir:/c\log_dir: $WK_DIR/mt-ag/log" $WK_DIR/$MT_AG_CONFIG_FILE
sed -i "/^data_dir:/c\data_dir: $APE_DATA_DIR" $WK_DIR/$MT_AG_CONFIG_FILE
sed -i "/^embedding_output_dir:/c\embedding_output_dir: $WK_DIR" $WK_DIR/$MT_AG_CONFIG_FILE

cd $APE_SRC_PATH && python3 -m translate $WK_DIR/$MT_AG_CONFIG_FILE --train -v

echo "Export MT-AG embedding"
cd $APE_SRC_PATH && python3 -m translate $WK_DIR/$MT_AG_CONFIG_FILE --train --save-embedding -v 

echo "STEP 3: Train LM with initial embedding"
MT_EMB_PATH=$WK_DIR"/embedding_mt:0.txt"
cd $LM_SRC_PATH && python3 ptb_word_lm.py --data_path=$LM_DATA_DIR --model ape --num_gpus=1 --save_path=$WK_DIR"/lm-mt/model"  --vocab_path=$LM_VOCAB_PATH --embedding_path=$MT_EMB_PATH --data_ext=de

echo "STEP 4: Train MT+AG+LM warm start"
MT_AG_LM_CONFIG_FILE="config-MT-AG-LM-en-de.yaml"
RNN_MODEL_DIR=$WK_DIR"/lm-mt/model"
RNN_CELL_NAME="Model/RNN/multi_rnn_cell/cell_0/basic_lstm_cell"
cp $APE_CONF_PATH/chained-syn.yaml $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "/^model_dir:/c\model_dir: $WK_DIR/mt-ag-lm/model" $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "/^log_dir:/c\log_dir: $WK_DIR/mt-ag-lm/log" $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "/^data_dir:/c\data_dir: $APE_DATA_DIR" $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "/^rnn_lm_model_dir:/c\rnn_lm_model_dir: $RNN_MODEL_DIR" $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "/^rnn_lm_cell_name:/c\rnn_lm_cell_name: $RNN_CELL_NAME" $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "0,/embedding_file:/s/embedding_file/c\    embedding_file: $MT_EMB_PATH" $WK_DIR/$MT_AG_LM_CONFIG_FILE
sed -i "/origin_model_ckpt:/c\origin_model_ckpt: $WK_DIR/mt-ag/model" $WK_DIR/$MT_AG_LM_CONFIG_FILE
cd $APE_SRC_PATH && python3 -m translate $WK_DIR/$MT_AG_LM_CONFIG_FILE --train -v

echo "FINISH TRAIN"

