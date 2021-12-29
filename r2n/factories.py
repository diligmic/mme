import tensorflow as tf

from r2n.utils import StringEqualsCaseInsensitive
import r2n.utils as utils
import r2n.knowledge as knowledge
import r2n.layers as layers
import r2n.kge as kge

############################################
"""Construct a KGE"""
def kge_factory(
    inputs: dict,  # inputs (dict of tf.keras.Input layers of shape (num_constants, num_constants))
    ontology: knowledge.Ontology,
    params: utils.Params):

    # Input is dict domain -> tensor
    # Constants is dict domain -> tensor
    # Atoms are serialized HB tensor

    ##############
    # Constant embedders
    constant_embedding_enabled, _, _ = utils.get_constant_embedding_size_by_domain(
        inputs, ontology.domains.keys(), params)

    constant_embedding_sizes = None
    if constant_embedding_enabled:
        if isinstance(inputs, dict):
            constant_embedding_layer = layers.ConstantEmbedding(
                inputs, ontology, params
            )
        else:
            assert (
                params.constant_embedding_sizes
                and params.constant_embedding_sizes[-1] > 0
            )
            constant_embedding_sizes = params.constant_embedding_sizes[-1]
            constant_embedding_layer = layers.DomainConstantEmbedding(
                ontology, params.constant_embedding_sizes[-1], params
            )
        constant_embedding_layer._name = "constant_embedder"
    else:
        constant_embedding_layer = None

    ##############
    # Atom embedders
    atom_embedder_enabled = (
        params.atom_embedding_size
        and params.atom_embedder
    )
    if atom_embedder_enabled:
        params.atom_embedding_size = params.atom_embedding_size[-1]
    elif constant_embedding_enabled:
        params.atom_embedding_size = constant_embedding_sizes
    else:
        params.atom_embedding_size = -1

    if atom_embedder_enabled:
        atom_embedding_layer = layers.AtomEmbeddingLayer(
            inputs, ontology, params)

        if StringEqualsCaseInsensitive(params.atom_embedder, 'TransE'):
            output_layer = kge.TransEAtomOutputLayer()

        elif StringEqualsCaseInsensitive(params.atom_embedder, 'RotateE'):
            output_layer = kge.RotateEAtomOutputLayer()

        elif StringEqualsCaseInsensitive(params.atom_embedder, 'DistMult'):
            output_layer = kge.DistMultAtomOutputLayer()

        elif StringEqualsCaseInsensitive(params.atom_embedder, 'ComplEx'):
            output_layer = kge.ComplexAtomOutputLayer()

        elif StringEqualsCaseInsensitive(params.atom_embedder, 'NTN'):
            output_layer = kge.NeuralTensorNetworkAtomOutputLayer(
                ontology, params.atom_embedding_size,
                activation=params.atom_embedding_activation)

        elif StringEqualsCaseInsensitive(params.atom_embedder, 'MLP'):
            output_layer = kge.MLPAtomOutputLayer(
                reg_weight=params.atom_embedder_regularization,
                activation=params.atom_embedding_activation)

        elif StringEqualsCaseInsensitive(params.atom_embedder, 'MLP_cossim'):
            output_layer = kge.CosineSimAtomOutputLayer(params.atom_embedding_size)

        else:
            assert not params.atom_embedder, (
                'Unknown embedder %s' % params.atom_embedder)

        atom_embedding_layer._name = 'atom_embedder'
        output_layer._name = 'no_reasoning_output'

    else:
        atom_embedding_layer = None
        output_layer = None

    ##############
    # Assemble layers.
    return (constant_embedding_layer,
            atom_embedding_layer,
            output_layer)

def layers_factory(
    inputs: dict,  # inputs (dict of tf.keras.Input layers of shape (num_constants, num_constants))
    ontology: knowledge.Ontology,
    grounded_formulas: knowledge.GroundedFormula,
    params: utils.Params):

    (constant_embedding_layer,
     atom_embedding_layer,
     no_reasoning_output_layer) = kge_factory(inputs, ontology, params)

    if atom_embedding_layer is None and params.reasoning_on_embeddings:
        print("Setting the atom embedding size to:", inputs.shape[-1])
        params.atom_embedding_sizes = [inputs.shape[-1]]
    elif not params.reasoning_on_embeddings:
        params.atom_embedding_sizes = [1]

    # Formula embedders
    if params.formula_embedder is not None and utils.string_equals_case_insensitive(params.formula_embedder, "Transformers"):
        reasoning_layer = layers.TransformersBasedFormulaEmbeddingLayer(
            ontology, grounded_formulas, params
        )
        reasoning_output_layer = layers.TransformersBasedFormulaOutputLayer()
    elif params.formula_embedder is not None and StringEqualsCaseInsensitive(params.formula_embedder, "FGNN"):
        params.transformer_masked_predictions = True
        reasoning_layer = layers.FGNNBasedFormulaEmbeddingLayer(
            ontology, grounded_formulas, params)
        reasoning_output_layer = layers.TransformersBasedFormulaOutputLayer()
    elif params.formula_embedder is not None or params.formula_embedder != "":
        assert not params.formula_embedder, (
            "Unknown reasoning type %s" % params.formula_embedder
        )
        reasoning_layer = reasoning_output_layer = None
    else:
        reasoning_layer = reasoning_output_layer = None

    if reasoning_layer:
        reasoning_layer._name = "reasoning_embedder"
    if reasoning_output_layer:
        reasoning_output_layer._name = "reasoning_output"

    # assert not (
    #     params.semantic_ground_formulas and params.semantic_on_output
    # ), f"""params.semantic_ground_formulas={params.semantic_ground_formulas}\
    #         params.semantic_on_output={params.semantic_on_output}"""
    # if params.semantic_ground_formulas:
    if utils.string_equals_case_insensitive(params.model, "R2NS"):
        cliques_layer = layers.CliquesOutputLayer(ontology, grounded_formulas, params)
    # elif params.semantic_on_output:
    elif utils.string_equals_case_insensitive(params.model, "R2NSO"):
        cliques_layer = layers.SemanticOutputLayer(ontology, grounded_formulas)
    else:
        cliques_layer = None

    return (
        constant_embedding_layer,
        atom_embedding_layer,
        no_reasoning_output_layer,
        reasoning_layer,
        reasoning_output_layer,
        cliques_layer,
    )


def no_reasoning_model_factory(
    inputs,
    constant_embedding_layer,
    atom_embedding_layer,
    no_reasoning_output_layer,
    loss,
    metrics,
    params,
):
    if constant_embedding_layer:
        constants = constant_embedding_layer(inputs)
    else:
        constants = inputs

    atom_embeddings = atom_embedding_layer(constants)
    predictions = no_reasoning_output_layer(atom_embeddings)
    model = tf.keras.models.Model(
        inputs=inputs, outputs=predictions, name="model_no_reasoning"
    )
    optimizer = utils.get_optimizer(params.optimizer)
    model.compile(
        optimizer=optimizer(params.pretrain_learning_rate),
        loss=loss,
        run_eagerly=params.debug,
        metrics=metrics,
    )
    model.summary(line_length=150)
    return model


def model_factory(
    inputs,
    constant_embedding_layer,
    atom_embedding_layer,
    reasoning_layer,
    reasoning_output_layer,
    cliques_layer,
    loss,
    metrics,
    params,
):

    # Embed the constant or use them as input for the next layer
    if constant_embedding_layer:
        constants = constant_embedding_layer(inputs)
    else:
        constants = inputs

    # Embed the atoms (KGE) starting from the constants
    if atom_embedding_layer:
        atom_embeddings = atom_embedding_layer(constants)
    else:
        atom_embeddings = constants

    # Compute the atom predictions with reasoning
    if reasoning_layer:
        atom_embeddings_after_reasoning = reasoning_layer(atom_embeddings)

        if reasoning_output_layer:
            atom_predictions = reasoning_output_layer(atom_embeddings_after_reasoning)
        else:
            atom_predictions = atom_embeddings_after_reasoning
    else:
        atom_predictions = None

    # Compute cliques predictions
    if cliques_layer:
        if utils.string_equals_case_insensitive(params.model, "R2NS"):
            clique_predictions = cliques_layer(atom_embeddings_after_reasoning)
        elif utils.string_equals_case_insensitive(params.model, "R2NSO"):
            clique_predictions = cliques_layer(atom_predictions)
        else:
            clique_predictions = None

        if clique_predictions is not None:
            predictions = tf.concat([atom_predictions, clique_predictions], axis=1)
    else:
        predictions = atom_predictions

    metrics = [metrics]

    model = tf.keras.models.Model(inputs=inputs, outputs=predictions, name="output")
    optimizer = utils.get_optimizer(params.optimizer)
    model.compile(
        optimizer=optimizer(params.learning_rate),
        loss=loss,
        run_eagerly=params.debug,
        metrics=metrics,
    )
    model.summary(line_length=150)
    return model

def model_factory_fgnn(
    inputs,
    constant_embedding_layer,
    atom_embedding_layer,
    no_reasoning_output_layer,
    reasoning_layer,
    reasoning_output_layer,
    loss,
    metrics,
    params,
):

    # Embed the constant or use them as input for the next layer
    constants = constant_embedding_layer(inputs)
    atom_embeddings = atom_embedding_layer(constants)
    no_reasoning_predictions = no_reasoning_output_layer(atom_embeddings)

    atom_embeddings_after_reasoning = reasoning_layer(
        tf.expand_dims(no_reasoning_predictions, axis=-1))
    predictions = reasoning_output_layer(atom_embeddings_after_reasoning)

    model = tf.keras.models.Model(
        inputs=inputs, outputs=[predictions, no_reasoning_predictions], name="output")
    optimizer = utils.get_optimizer(params.optimizer)
    model.compile(
        optimizer=optimizer(params.learning_rate),
        loss=[loss, loss],
        loss_weights=[10.0, 1.0],
        run_eagerly=params.debug,
        metrics=[metrics, metrics],
    )
    model.summary(line_length=150)
    return model
