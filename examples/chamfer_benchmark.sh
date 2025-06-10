#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=jereminuer/qwen25_vl_3b_sft
MODEL_PATH=afland/mentis-qwen2.5-vl-3b-grpo-step80

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=shalunov/fusion360-normal-map.parquet[0%:1%] \
    data.val_files=shalunov/fusion360-normal-map.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_cadquery_no_think_grpo \
    trainer.n_gpus_per_node=2

# Need to run this to get cadquery to work in EasyR1 docker:
# apt-get install -y libgl1-mesa-glx

# More commands:
# sudo apt-get install -y python3.10-venv
# python3.10 -m venv myenv
# source myenv/bin/activate

# May need this too:
# apt-get install -y python3-dev python3.10-dev build-essential