#!/bin/bash

# cd ~/Freebase-Setup
# python3 virtuoso.py start 3001 -d virtuoso_db
# cd ~/KB-BINDER

# HF_ENDPOINT=https://hf-mirror.com python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
#  --api_key key --engine Qwen/Qwen2.5-32B-Instruct \
#  --train_data_path ./data/train.json --eva_data_path ./data/dev.json \
#  --fb_roles_path ./data/fb_roles --surface_map_path ./data/surface_map_file_freebase_complete_all_mention

# HF_ENDPOINT=https://hf-mirror.com \
# python3 test_kbqa.py --shot_num 80 --temperature 0.3 \
#     --api_key key --engine Qwen/Qwen2.5-32B-Instruct \
#     --train_data_path data/processed_spice_data/train_each_type_10.json --eva_data_path data/processed_spice_data/dev_each_type_5.json \
#     --exp_name train10_dev5_shot80

HF_ENDPOINT=https://hf-mirror.com \
python3 baseline_top1.py --shot_num 40 \
    --train_data_path data/processed_spice_data/train_each_type_50.json \
    --eva_data_path data/processed_spice_data/dev_each_type_50.json