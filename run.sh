#!/bin/bash

# cd ~/Freebase-Setup
# python3 virtuoso.py start 3001 -d virtuoso_db
# cd ~/KB-BINDER

# HF_ENDPOINT=https://hf-mirror.com python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
#  --api_key key --engine Qwen/Qwen2.5-32B-Instruct \
#  --train_data_path ./data/train.json --eva_data_path ./data/dev.json \
#  --fb_roles_path ./data/fb_roles --surface_map_path ./data/surface_map_file_freebase_complete_all_mention

# HF_ENDPOINT=https://hf-mirror.com \
# python3 test_kbqa.py --shot_num 40 --temperature 0.3 \
#     --api_key key --engine Qwen/Qwen2.5-32B-Instruct \
#     --train_data_path data/processed_spice_data/train_1000.json --eva_data_path data/processed_spice_data/dev_1000.json \
#     --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention \
#     --freebase_map_path data/freebase_mid_to_fn_dict.pickle --wikidata_map_path data/wikidata_mid_to_fn_dict.pickle

HF_ENDPOINT=https://hf-mirror.com \
python3 baseline.py --shot_num 40 --temperature 0.3 \
    --api_key sk-54964e5c3b8c4998a74f7d3e35b618ac --engine deepseek-chat \
    --train_data_path data/processed_spice_data/train_each_type_50.json --eva_data_path data/processed_spice_data/dev_each_type_50.json \
    --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention \
    --freebase_map_path data/freebase_mid_to_fn_dict.pickle --wikidata_map_path data/wikidata_mid_to_fn_dict.pickle