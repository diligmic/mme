# Relational Reasoning Networks (R2N)

## Setup and first launch
1. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Launch the script with `-h` flag to help: 
    ```
    ./bin/run.sh -h
    ```

    It should appear something like this:
    ```
    R2N - Run the relational reasoning model for the countries dataset.

    Usage:
    run.sh [options]
    run.sh -m "R2NC R2NS R2NSO" -e 300 -r 10 -t "S1 S2 S3"

    Options:
    -m, --model <M>                 Run the model(s) (R2NC, R2NS, R2NSO) [R2NC R2NS R2NSO]
    -e, --epochs <E>                Number of epochs [300]
    -r, --runs <R>                  Number of runs [10]
    -s, --tasks <S>                 Tasks for the countries dataset [S1 S2 S3]
    -t, --num_transformers <T>      Number of chained transformers (multi-hops) [3]
    -a, --atom_embedding_size <A>   Atom embedding size [30]
    -p, --pretrain                  Set pretrain to true.

    -h, --help      Display this help and exit
    ```

## Models
The Relational Reasoning Network comes with three different models (update 02 Dec 2021):
* `R2NC`: Relational Reasoning Network with Correlation.
* `R2NS`: Relational Reasoning Network with Semantic.
* `R2NSO`: Relational Reasoning Network with Semantic on Output. 

Other models are currently under development.

### R2NC
The `R2NC` model is the original one and exploits hidden correlation among entities. For instance, given the fact that three entities A, B, and C are correlated by any kind of logical formula, the model uses this kind of information to positionally concatenate the embeddings of the entities and reasons over this newly created concatenated embedding. No semantic is injected in this model, the network is free to discover any hidden correlation.

### R2NS
The `R2NS` model injects the semantic of the FOL formulas by concatenating the output of the atom embeddings for any given clique and by supervising these embeddings with the truth value of the clique to which they refer to. In this model, the semantic is forced at clique level but not operators are directly involved.

### R2NSO
The `R2NSO` model injects the semantic of the FOL formulas directly on the output of the atom embeddings. The atom embeddings are mapped to real values and the logical operators are applied to those values according to a given logic (i.e. Lukasiewicz Logic or Product Logic). The result is supervised according to the truth value of the clique to which the atoms refer to.