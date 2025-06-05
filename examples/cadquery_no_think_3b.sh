#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=jereminuer/qwen25_vl_3b_sft

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=shalunov/mentis-public-rl-cad-recode@train[0%:90%] \
    data.val_files=shalunov/mentis-public-rl-cad-recode@train[90%:100%] \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_cadquery_no_think_grpo \
    trainer.n_gpus_per_node=1

# Need to run this to get cadquery to work in EasyR1 docker:
# apt-get install -y libgl1-mesa-glx