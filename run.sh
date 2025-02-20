#!/bin/bash

# cd ~/Freebase-Setup
# python3 virtuoso.py start 3001 -d virtuoso_db
# cd ~/KB-BINDER

# HF_ENDPOINT=https://hf-mirror.com python3 few_shot_kbqa.py --shot_num 40 --temperature 0.3 \
#  --api_key key --engine Qwen/Qwen2.5-32B-Instruct \
#  --train_data_path ./data/train.json --eva_data_path ./data/dev.json \
#  --fb_roles_path ./data/fb_roles --surface_map_path ./data/surface_map_file_freebase_complete_all_mention

HF_ENDPOINT=https://hf-mirror.com \
python3 few_shot_kbqa_for_spice.py --shot_num 40 --temperature 0.3 \
    --api_key key --engine Qwen/Qwen2.5-32B-Instruct \
    --train_data_path data/processed_spice_data/train_simple.json --eva_data_path data/processed_spice_data/dev_simple.json \
    --fb_roles_path data/fb_roles --surface_map_path data/surface_map_file_freebase_complete_all_mention \
    --freebase_map_path data/freebase_mid_to_fn_dict.pickle --wikidata_map_path data/wikidata_mid_to_fn_dict.pickle