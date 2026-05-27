# TAG variant: FrameSamp + modulation + temporal encoder + memory gate
# Baseline to beat: perceptual-framesamp-modul

MME_VLA_TYPE="perceptual-framesamp-tag"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py mme_vla_suite \
--exp-name=${MME_VLA_TYPE} \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_data \
--model.use_history \
--model.history_config="${MME_VLA_TYPE}.yaml"
