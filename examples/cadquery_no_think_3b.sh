#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=jereminuer/qwen25_vl_3b_sft

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=shalunov/test@train[0:1000] \
    data.val_files=shalunov/test@train[0:500] \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_cadquery_no_think_grpo \
    trainer.n_gpus_per_node=1