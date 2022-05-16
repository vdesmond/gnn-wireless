#!/bin/bash

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S" # current time
}

# Runs for evaluating D2D
# Train: 1000 Valid: 500 Test: 500
# Hidden state dimension: 16
d2d_sweep() {
    arr=(10 20 30 40 50)
    for i in ${!arr[@]}; do
        echo "python gen_dataset.py -t 1000 -v 500 -s 500 -d ${arr[i]}"
        echo "python main.py >> d2d_${arr[i]}_log_$(timestamp).txt"
    done
}

hidden_state_dim_sweep() {
    arr=(8 12 16 24 32)
    # Simulation for 20 D2D pairs
    echo "python gen_dataset.py -t 1000 -v 500 -s 500 -d 20"
    for i in ${!arr[@]}; do
        echo "yaml-set --change=hidden_state_dimension --value="${arr[i]}" global_variables.yaml"
        echo "python main.py >> hidden_state_dim_${arr[i]}_log_$(timestamp).txt"
    done
    # Reset to default hidden state dimension (16)
    echo "yaml-set --change=hidden_state_dimension --value="16" global_variables.yaml"
}

# MAIN
#hidden_state_dim_sweep
