## {1}: config file path
## {2}: gpu number
## {3}: output dir path, only used when resume training

# export DETECTRON2_DATASETS=/data/disk1/datasets

if [ -f "${3}/last_checkpoint" ]; then
    echo "Resume training from output/last_checkpoint"
    python3 train_net.py \
        --dist-url "auto" \
        --num-gpus ${2} \
        --config-file ${1} \
        --resume
else
    echo "Train from begining"
    python3 train_net.py \
        --dist-url "auto" \
        --num-gpus ${2} \
        --config-file ${1}
fi
