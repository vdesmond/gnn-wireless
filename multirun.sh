#!/bin/bash

# Define test constants
TRAIN=200
VAL=200
TEST=1000
GEN_DATASET=20 # For tests other than d2d

main() {
    #d2d_sweep
    #hidden_state_dim_sweep
    #aggregation_sweep
    #number_of_iterations_sweep
    #training_size_sweep
}

#? FUNCTION DEFINITIONS

timestamp() {
    date +"%Y_%m_%d_%H_%M_%S" # current time
}

gen_dataset() {
    python gen_dataset.py -t $TRAIN -v $VAL -s $TEST -d $1
}

yaml_set() {
    yaml-set --change=$1 --value="$2" $3.yaml
}

yaml_del() {
    yaml-set --delete --change=$1 $2.yaml
}

log_process() {
    dos2unix -c Mac -f $1_$2_$3_t.txt
    if [[ $VAR -gt 10 ]]; then
        INC=$TRAIN
    else
        INC=$TEST
    fi
    grep -n "$INC/$INC" $1_$2_$3_t.txt | grep -v "ETA" >$1_$2_$3.txt
    rm $1_$2_$3_t.txt
}

run_main() {

    # Timestamp
    T=$3
    temp=${T:0:-8}

    # Train and validate
    yaml_set validation_dataset ./data/val train_options
    yaml_del load_model_path train_options
    python main.py | tee $1_$2_log_t.txt
    log_process $1 $2 log

    # Rename CheckPoint folder
    mv -T ./CheckPoint/experiment_$temp* ./CheckPoint/$1_$2_expt_$3

    # Evaluate
    P=$(ls -d ./CheckPoint/$1_$2_expt_$3/ckpt/* | tail -1)
    yaml_set validation_dataset ./data/test train_options
    yaml_set load_model_path $P train_options
    python evaluate.py | tee $1_$2_evaluation_t.txt
    log_process $1 $2 evaluation

    # Store log, eval and config
    mv ./$1_$2_log.txt ./$1_$2_evaluation.txt ./CheckPoint/$1_$2_expt_$3
    cat model_description.yaml train_options.yaml global_variables.yaml >./CheckPoint/$1_$2_expt_$3/$1_$2_config.txt
}

d2d_sweep() { # Runs for evaluating D2D
    TS=$(timestamp)
    arr=(10 20 30 40 50)
    for i in ${!arr[@]}; do
        gen_dataset ${arr[i]}
        run_main d2d ${arr[i]} $TS
    done
}

hidden_state_dim_sweep() {
    arr=(8 16 32 48 64)
    # Simulation for 20 D2D pairs
    gen_dataset $GEN_DATASET
    TS=$(timestamp)
    for i in ${!arr[@]}; do
        yaml_set hidden_state_dimension ${arr[i]} global_variables
        run_main hidden_state_dim ${arr[i]} $TS
    done
    # Reset to default hidden state dimension (16)
    yaml_set hidden_state_dimension 16 global_variables
}

aggregation_sweep() {
    arr=(min max mean convolution attention)
    # Simulation for 20 D2D pairs
    gen_dataset $GEN_DATASET
    TS=$(timestamp)
    for i in ${!arr[@]}; do
        yaml_set message_passing.stages.stage_message_passings.aggregation.type ${arr[i]} model_description
        run_main aggregation ${arr[i]} $TS
    done
    # Reset to default aggregation (attention)
    yaml_set message_passing.stages.stage_message_passings.aggregation.type attention model_description
}

number_of_iterations_sweep() {
    arr=(2 3 4 5)
    # Simulation for 20 D2D pairs
    gen_dataset $GEN_DATASET
    TS=$(timestamp)
    for i in ${!arr[@]}; do
        yaml_set message_passing.num_iterations ${arr[i]} model_description
        run_main hidden_state_dim ${arr[i]} $TS
    done
    # Reset to default hidden state dimension (16)
    yaml_set message_passing.num_iterations 3 model_description
}

training_size_sweep() { # Runs for evaluating D2D
    # Train: 1000 Valid: 500 Test: 1000
    # Hidden state dimension: 16
    TS=$(timestamp)
    arr=(200 500)
    for i in ${!arr[@]}; do
        TRAIN=${arr[i]}
        gen_dataset $GEN_DATASET
        yaml_set epoch_size ${arr[i]} train_options
        run_main training ${arr[i]} $TS
    done
}

main "$@"
