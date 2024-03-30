#!/bin/bash

# Configure parameters here.
embedding_size=128
hidden_dim=128
n_glimpses=1
tanh_exploration=10
use_tanh=false
beta=0.9
max_grad_norm=2.0
learning_rate=3e-4
attention="BHD"
training_samples=100000
validation_samples=1000
n_epochs=5
num_nodes=16

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedding_size) embedding_size="$2"; shift 2;;
        --hidden_dim) hidden_dim="$2"; shift 2;;
        --n_glimpses) n_glimpses="$2"; shift 2;;
        --tanh_exploration) tanh_exploration="$2"; shift 2;;
        --use_tanh) use_tanh=true; shift;;
        --beta) beta="$2"; shift 2;;
        --max_grad_norm) max_grad_norm="$2"; shift 2;;
        --learning_rate) learning_rate="$2"; shift 2;;
        --attention) attention="$2"; shift 2;;
        --training_samples) training_samples="$2"; shift 2;;
        --validation_samples) validation_samples="$2"; shift 2;;
        --n_epochs) n_epochs="$2"; shift 2;;
        --num_nodes) num_nodes="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Run Python script with arguments
python ../run_cbrl_train.py \
    --embedding_size "$embedding_size" \
    --hidden_dim "$hidden_dim" \
    --n_glimpses "$n_glimpses" \
    --tanh_exploration "$tanh_exploration" \
    $(if [ "$use_tanh" = true ]; then echo "--use_tanh"; fi) \
    --beta "$beta" \
    --max_grad_norm "$max_grad_norm" \
    --learning_rate "$learning_rate" \
    --attention "$attention" \
    --training_samples "$training_samples" \
    --validation_samples "$validation_samples" \
    --n_epochs "$n_epochs" \
    --num_nodes "$num_nodes"
