#!/bin/bash

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S" # current time
}

gen_dataset() {
    echo "python gen_dataset.py -t 1000 -v 500 -s 500 -d $1"
}

run_main() {
    echo "python main.py >> $1_$2_log_$3.txt"
}

yaml_set() {
    echo "yaml-set --change=$1 --value="$2" $3.yaml"
}

d2d_sweep() { # Runs for evaluating D2D
    # Train: 1000 Valid: 500 Test: 500
    # Hidden state dimension: 16
    TS=$(timestamp)
    arr=(10 20 30 40 50)
    for i in ${!arr[@]}; do
        gen_dataset ${arr[i]}
        run_main d2d ${arr[i]} $TS
    done
}

hidden_state_dim_sweep() {
    arr=(8 12 16 24 32)
    # Simulation for 20 D2D pairs
    gen_dataset 20
    TS=$(timestamp)
    for i in ${!arr[@]}; do
        yaml_set hidden_state_dimension ${arr[i]} global_variables
        run_main hidden_state_dim ${arr[i]} $TS
    done
    # Reset to default hidden state dimension (16)
    yaml_set hidden_state_dimension 16 global_variables
}

aggregation_sweep() {
    arr=(sum min max mean convolution attention)
    # Simulation for 20 D2D pairs
    gen_dataset 20
    TS=$(timestamp)
    for i in ${!arr[@]}; do
        yaml_set message_passing.stages.stage_message_passings.aggregation.type ${arr[i]} model_description
        run_main aggregation ${arr[i]} $TS
    done
    # Reset to default aggregation (attention)
    yaml_set message_passing.stages.stage_message_passings.aggregation.type attention model_description
}

# MAIN
#d2d_sweep
#hidden_state_dim_sweep
#aggregation_sweep
