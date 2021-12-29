#!/bin/bash

RUNS=10
TASKS="S2"
MODEL="R2NC"
PRETRAIN_EPOCHS=30
PRETRAIN_LR=0.03
EPOCHS=500
LR=0.03
ATOM_EMBEDDER="ComplEx"
ATOM_EMBEDDING_SIZE=30
NUM_TRANSFORMERS=1
FORMULA_FILE="data/countries/formulas.txt"
SEED=1000

DATE=0000000002
ODIR=./results/countries_fgnn1

options=$(getopt -n $0 -o hm:r:t:pe:l:a:T: \
 -l help,model:,runs:,tasks:,pretrain,pretrain_epochs:,pretrain_learning_rate: \
 -l epochs:,learning_rate:,atom_embedding_size:,num_transformers: \
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
\032\032 -E, --pretrain_epochs <PRETRAIN_EPOCHS>\t Number of pretrain epochs [$PRETRAIN_EPOCHS]\n\
\032\032 -L, --pretrain_learning_rate <PRETRAIN_LR>\t Pretrain learning rate [$PRETRAIN_LR]\n\
\032\032 -e, --epochs <EPOCHS>\t\t\t\t Number of epochs [$EPOCHS]\n\
\032\032 -l, --learning_rate <LR>\t\t\t Learning rate [$LR]\n\
\032\032 -T, --num_transformers <NUM_TRANSFORMERS>\t Number of chained transformers (multi-hops) [$NUM_TRANSFORMERS]\n\
\032\032 -a, --atom_embedding_size <ATOM_EMBEDDING_SIZE> Atom embedding size [$ATOM_EMBEDDING_SIZE]\n\
\032\032 --formula_file <FORMULA_FILE>\t\t\t File containing the logical formulas [$FORMULA_FILE]\n\n\
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
    -E|--pretrain_epochs)
        PRETRAIN_EPOCHS=$2; shift; ;;
    --L|--pretrain_learning_rate)
        PRETRAIN_LR=$2; shift; ;;
    -e|--epochs)
        EPOCHS=$2; shift; ;;
    -l|--learning_rate)
        LR=$2; shift; ;;
    -a|--atom_embedding_size)
        ATOM_EMBEDDING_SIZE=$2; shift; ;;
    -T|--num_transformers)
        NUM_TRANSFORMERS=$2; shift; ;;
    --formula_file)
        FORMULA_FILE=$2; shift; ;;
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
echo "PRETRAIN_EPOCHS: $PRETRAIN_EPOCHS"
echo "PRETRAIN_LR: $PRETRAIN_LR"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "ATOM_EMBEDDING_SIZE: $ATOM_EMBEDDING_SIZE"
echo "ATOM_EMBEDDER: $ATOM_EMBEDDER"
echo "NUM_TRANSFORMERS: $NUM_TRANSFORMERS"
echo "FORMULA_FILE: $FORMULA_FILE"

for S in $TASKS; do
    for M in $MODEL; do
        mkdir -p ${ODIR}/${M}
        for A in $ATOM_EMBEDDING_SIZE; do
          for AE in $ATOM_EMBEDDER; do
            for T in $NUM_TRANSFORMERS; do
              F=$ODIR/${M}/${S}_A${A}_AE${AE}_T${T}_E${EPOCHS}_R${RUNS}_$DATE.txt
              echo "S $S | M $M | A $A | AE $AE | T $T | E $EPOCHS | R $RUNS" | tee -a $F
              EXP_NUM=0
              for R in $(seq 1 $RUNS); do
                echo -e "\nR $R"
                set -x
                python3 r2n_fgnn.py \
                            --model $M \
                            --task $S \
                            --pretrain \
                            --pretrain_epochs $PRETRAIN_EPOCHS \
                            --pretrain_learning_rate $PRETRAIN_LR \
                            --epochs $EPOCHS \
                            --learning_rate $LR \
                            --atom_embedding_size $A \
                            --atom_embedder ${AE} \
                            --formula_embedder "FGNN" \
                            --transformer_embedding_size $A \
                            --num_transformers $T \
                            --formula_file $FORMULA_FILE \
                            --seed=$((SEED+EXP_NUM))
                        set +x
                        EXP_NUM=$((EXP_NUM+1))
                    done | tee -a $F
                    echo -e "\n" | tee -a $F
                    grep 'Accuracy_with\|AUC_with' ${F} | . ./bin/avg_std.sh | tee -a ${F}
                    echo -e "\n\n" | tee -a $F
                done
            done
          done
        done
    done
done


# Get Results
for S in $TASKS; do
    for M in $MODEL; do
      for A in $ATOM_EMBEDDING_SIZE; do
        for AE in $ATOM_EMBEDDER; do
            for T in $NUM_TRANSFORMERS; do
              F=$ODIR/${M}/${S}_A${A}_AE${AE}_T${T}_E${EPOCHS}_R${RUNS}_$DATE.txt
              if [[ -e $F ]]; then
                echo "S $S | M $M | A $A | AE${AE} | T $T | E $EPOCHS | R $RUNS" | tee -a $ODIR/$DATE.txt
                grep 'Accuracy_with\|AUC_with' ${F} | . ./bin/avg_std.sh | tee -a $ODIR/$DATE.txt
                echo | tee -a $ODIR/$DATE.txt
              fi
            done
        done
      done
    done
done
