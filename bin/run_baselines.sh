#!/bin/bash

MAX_JOBS=2
RUNS=10
TASKS="S2"
EPOCHS=100
LR=0.01
MODEL="ComplEx DistMult"  # TransE  ComplEx 
ATOM_EMBEDDING_SIZE="30"
SEED=1000

options=$(getopt -n $0 -o hm:r:t:pe:l:a:T: \
 -l help,model:,runs:,tasks: \
 -l epochs:,pretrain_learning_rate:,atom_embedding_size:,num_transformers: \
 -l formula_file: \
 -- "$@")

[ $? -eq 0 ] || {
    echo "Incorrect options provided"
    exit 1
}

eval set -- "$options"

usage="\n\
R2N - Run the relational reasoning model for the countries dataset.\n\n\
Usage:\n\
\032\032 $(basename "$0") [options]\n\
\032\032 $(basename "$0") -m \"$MODEL\" -e $EPOCHS -r $RUNS -t \"$TASKS\"\n\
\n\
Options:\n\
\032\032 -m, --model <MODEL>\t\t\t\t Run the model(s) (R2NC, R2NS, R2NSO) [$MODEL]\n\
\032\032 -r, --runs <RUNS \t\t\t\t Number of runs [$RUNS]\n\
\032\032 -t, --tasks <TASKS>\t\t\t\t Tasks for the countries dataset [$TASKS]\n\
\032\032 -e, --epochs <EPOCHS>\t\t\t\t Number of epochs [$EPOCHS]\n\
\032\032 -l, --learning_rate <LR>\t\t\t Learning rate [$LR]\n\
\032\032 -a, --atom_embedding_size <ATOM_EMBEDDING_SIZE> Atom embedding size [$ATOM_EMBEDDING_SIZE]\n\
\032\032 -h, --help \t Display this help and exit\n"

while true; do
    case "$1" in
    -h|--help)
        echo -e $usage || xxd -p -r; exit 2;;
    -m|--model)
        MODEL=$2; shift; ;;
    -r|--runs)
        RUNS=$2; shift; ;;
    -t|--tasks)
        TASKS=$2; shift; ;;
    -e|--epochs)
        EPOCHS=$2; shift; ;;
    -l|--learning_rate)
        LR=$2; shift; ;;
    -a|--atom_embedding_size)
        ATOM_EMBEDDING_SIZE=$2; shift; ;;
    --)
        shift; break ;;
    *)
        echo "Invalid options!!";
        exit 1
        ;;
    esac
    shift
done

echo "MODEL: $MODEL"
echo "RUNS: $RUNS"
echo "TASKS: $TASKS"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "ATOM_EMBEDDING_SIZE: $ATOM_EMBEDDING_SIZE"

DATE=$(date +"%d%m%y_%H%M%S")
DATE="271221_234154"
ODIR=./results/countries_baselines

declare -A cur_jobs=( ) # build an associative array w/ PIDs of jobs we started

for S in $TASKS; do
    for M in $MODEL; do
        mkdir -p ${ODIR}/${M}
        for A in $ATOM_EMBEDDING_SIZE; do
            F=$ODIR/${M}/${S}_A${A}_E${EPOCHS}_R${RUNS}_$DATE.txt
            echo "S $S | M $M | A $A | E $EPOCHS | R $RUNS" | tee -a $F
            EXP_NUM=0
            for R in $(seq 1 $RUNS); do
              if (( ${#cur_jobs[@]} >= ${MAX_JOBS} )); then
                wait -n # wait for at least one job to exit
                # Remove jobs not running anymore.
                for pid in "${!cur_jobs[@]}"; do
                  kill -0 "$pid" 2>/dev/null && unset cur_jobs[$pid]
                done
              fi
              echo -e "\nR $R"
              set -x
              python3 baseline.py \
                            --task $S \
                            --epochs $EPOCHS \
                            --pretrain_learning_rate $LR \
                            --atom_embedding_size $A \
                            --atom_embedder $M \
                            --seed $((SEED+EXP_NUM)) & cur_jobs[$!]=1
              set +x
              EXP_NUM=$((EXP_NUM+1))
            done | tee -a $F
            echo -e "\n" | tee -a $F
            grep 'Accuracy_baseline\|AUC_baseline' ${F} | \
                . ./bin/avg_std.sh | tee -a ${F}
            echo -e "\n\n" | tee -a $F
        done
    done
done
wait

# Get Results
for S in $TASKS; do
  for M in $MODEL; do
    for A in $ATOM_EMBEDDING_SIZE; do
      F=$ODIR/${M}/${S}_A${A}_E${EPOCHS}_R${RUNS}_$DATE.txt
      if [[ -e $F ]]; then
        echo "S $S | M $M | A $A | E $EPOCHS | R $RUNS" | tee -a $ODIR/$DATE.txt
        grep 'Accuracy_baseline\|AUC_baseline' ${F} | \
            . ./bin/avg_std.sh | tee -a $ODIR/$DATE.txt
        echo | tee -a $ODIR/$DATE.txt
      fi
    done
  done
done
