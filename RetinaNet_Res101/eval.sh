## {1}: config file path
## {2}: model weights file path

# export DETECTRON2_DATASETS=/data/disk1/datasets

python3 train_net.py \
  --config-file ${1} \
  --eval-only MODEL.WEIGHTS ${2}
