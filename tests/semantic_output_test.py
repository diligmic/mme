import os
import sys
import tempfile

import tensorflow as tf

from r2n import knowledge, utils, losses, factories


params = utils.Params()
params.atom_embedder = "TransE"
params.formula_embedder = "Transformers"
params.constant_embedding_sizes = []
params.atom_embedding_sizes = [30]
params.pre_train_epochs = 5
params.epochs = 100
params.pretrain = True
params.debug = True
params.formula_weight = 1.0
params.formula_balance_weight = 0.1
params.semantic_on_output = True


opts, params = utils.parse_args(sys.argv, params)
print("PARAMS", params)
print("OPTS", opts)

task = opts["task"] if "task" in opts else "S1"
train_predicates = ["locatedIn"]
val_predicates = test_predicates = ["locatedIn"]
formula_file = (
    opts["formula_file"] if "formula_file" in opts else r"data/countries/formulas.txt"
)
path_countries = r"data/countries"
path_kb = os.path.join(path_countries, f"knowledge.nl")
path_task_kb = os.path.join(path_countries, f"{task}.nl")
path_val_constants = os.path.join(path_countries, f"validation.txt")
path_test_constants = os.path.join(path_countries, f"test.txt")
path_val_regions = path_test_regions = os.path.join(path_countries, f"regions.txt")


print("Building ontology and herbrand interpretation")
ontology = knowledge.Ontology.from_file(path_kb)
herbrand_interpretation = tf.expand_dims(ontology.mask_by_file(path_kb), axis=0)

import time

print("Building formulas")
grounded_formulas = utils.get_formula_from_csv(
    formula_file, ontology, herbrand_interpretation
)

atom_outputs = tf.random.uniform(shape=(1, ontology.linear_size()), minval=0, maxval=1)


outputs = []
for i, grounded_formula in enumerate(grounded_formulas):
    cliques_atom_embeddings = tf.gather(
        params=atom_outputs,
        indices=grounded_formula.grounding_indices,
        axis=-1,
    )

    outputs.append(grounded_formula.formula.evaluate(cliques_atom_embeddings))

outputs = tf.expand_dims(tf.concat(outputs, axis=-1), axis=0)
print(tf.where(outputs))
print(outputs)
