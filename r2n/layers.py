import tensorflow as tf

import r2n.kge as kge
import r2n.knowledge as knowledge
import r2n.ntn as ntn
import r2n.logic as logic
import r2n.utils as utils


class DomainConstantEmbedding(tf.keras.layers.Layer):
    """Takes an hidden representation of constants for a single domain and creates
    representation. Optionally, embeds the constants before concatenations (if
    len(embedding_sizes) > 0)."""

    def __init__(self, o, embedding_size, params):
        super().__init__()
        activation = params.constant_embedder_activation
        reg_weight = params.constant_embedder_regularization
        normalize = params.constant_embedding_normalization
        assert embedding_size > 0

        self.o = o
        self.normalize = normalize
        self.embedder = None

        regularizer = None
        if reg_weight > 0.0:
            regularizer = tf.keras.regularizers.l2(reg_weight)
        self.embedder = tf.keras.layers.Dense(
            embedding_size,
            use_bias=False,
            activation=activation,
            kernel_regularizer=regularizer,
        )

    def call(self, inputs, **kwargs):
        features = self.embedder(inputs)
        if self.normalize:
            features = tf.math.l2_normalize(features, axis=-1)
        return features


class ConstantEmbedding(tf.keras.layers.Layer):
    """Calls the constant embedders, differenciating the behavior of the single domains."""

    def __init__(self, inputs, ontology, params):
        super().__init__()
        (
            enabled,
            domain2size,
            domain2enabled,
        ) = utils.get_constant_embedding_size_by_domain(
            inputs, ontology.domains.keys(), params
        )
        assert enabled, "Not requires constant embedders initialization."
        self.domain2size, self.domain2enabled = domain2size, domain2enabled

        self.embedder = {}
        for name, _ in ontology.domains.items():
            assert name in domain2enabled
            if not domain2enabled[name]:
                continue
            self.embedder[name] = DomainConstantEmbedding(
                ontology, domain2size[name], params
            )

    def call(self, domain_inputs):
        domain_features = {}
        for name, d in domain_inputs.items():
            if self.domain2enabled[name]:
                domain_features[name] = self.embedder[name](d)
            else:
                domain_features[name] = d
        return domain_features


class AtomEmbeddingLayer(tf.keras.layers.Layer):
    """
    Layers embedding Atoms.
    All layers assume to have as input tuple_features = self._build_tuples(inputs)
    and return a tensor
    batch_dim=1 x o.linear_size=(num_atoms_predicate[0]+num_atoms_predicate[1]+...) x embedding_dim

    Common functionalities to all embedding layers.
    """

    def __init__(self, inputs, ontology, params):
        super().__init__()
        self.atom_embedding_size = params.atom_embedding_size
        if not isinstance(self.atom_embedding_size, list):
            self.atom_embedding_size = [self.atom_embedding_size]
        assert self.atom_embedding_size[-1] > 0

        self.ontology = ontology
        self.normalize = params.atom_embedding_normalization
        self.reg_weight = params.atom_embedder_regularization

        self.drop_relational_dim = False

        self.embedders = self._atom_embedder_factory(inputs, params)

    def _build_tuples(self, inputs):
        tuple_features = {}
        for domain, ids in self.ontology.tuple_indices.items():
            X = []  # Cartesian product of the domains.
            for i in range(ids.shape[1]):
                features = inputs[domain[i]]
                indices = ids[:, i]
                x_i_th_element_of_tuple = tf.gather(features, indices, axis=1)
                X.append(x_i_th_element_of_tuple)
                tuple_features[domain] = X
        return tuple_features


    def _atom_embedder_factory(self, inputs, params):
        embedders = {}
        if not self.atom_embedding_size or self.atom_embedding_size[-1] <= 0:
            return embedders

        atom_embedding_size = self.atom_embedding_size[-1]

        for _, predicate in self.ontology.predicates.items():
            constant_embedding_size = None
            for domain in predicate.domains:
                _, const_emb_size = utils.get_constant_embedding_size(
                    inputs, domain, params
                )
                if constant_embedding_size is not None:
                    assert constant_embedding_size == const_emb_size
                else:
                    constant_embedding_size = const_emb_size

            atom_embedder = (params.atom_embedder
                             if predicate.name not in params.per_predicate_atom_embedder
                             else params.per_predicate_atom_embedder[predicate.name])


            if utils.string_equals_case_insensitive(atom_embedder, "TransE"):
                embedders[predicate.name] = kge.TransEEmbedder(
                    predicate,
                    atom_embedding_size,
                    activation=None,
                    reg_weight=self.reg_weight,
                )

            elif utils.StringEqualsCaseInsensitive(atom_embedder, 'RotateE'):
                assert atom_embedding_size == constant_embedding_size, (
                    'RotateE: constant and atom embedding space must be the same %d != %d.' % (atom_embedding_size, constant_embedding_size))
                embedders[predicate.name] = kge.RotateEEmbedder(
                    atom_embedding_size, reg_weight=self.reg_weight)

            elif utils.StringEqualsCaseInsensitive(atom_embedder, 'DistMult'):
                assert atom_embedding_size == constant_embedding_size, (
                    'DistMult: constant and atom embedding space must be the same %d != %d.' % (atom_embedding_size, constant_embedding_size))
                embedders[predicate.name] = kge.DistMultEmbedder(
                    atom_embedding_size,
                    reg_weight=self.reg_weight)

            elif utils.StringEqualsCaseInsensitive(atom_embedder, 'ComplEx'):
                assert 2 * atom_embedding_size == constant_embedding_size, (
                    'ComplEx: constant embedding size must be 2x atom embedding one. %d-%d' % (
                        constant_embedding_size, atom_embedding_size))
                embedders[predicate.name] = kge.ComplexEmbedder(
                    atom_embedding_size,
                    reg_weight=self.reg_weight)

            elif utils.StringEqualsCaseInsensitive(atom_embedder, 'NTN'):
                self.drop_relational_dim = True
                assert constant_embedding_size > 0
                embedders[predicate.name] = ntn.NeuralTensorLayer(
                    constant_embedding_size,
                    atom_embedding_size,
                    predicate.arity,
                    reg_weight=self.reg_weight)

            elif utils.StringEqualsCaseInsensitive(atom_embedder, 'MLP'):
                embedders[predicate.name] = kge.MLPEmbedder(atom_embedding_size,
                                                reg_weight=self.reg_weight)

            elif utils.StringEqualsCaseInsensitive(atom_embedder, 'MLP_cossim'):
                embedders[predicate.name] = kge.MLPCossimEmbedder(
                    predicate.arity, atom_embedding_size,
                    reg_weight=self.reg_weight)

            else:
                assert False, 'Unknown embedder:%s' % atom_embedder

        return embedders


    def call(self, inputs, **kwargs):
        tuple_features = self._build_tuples(inputs)

        if self.drop_relational_dim:
            for k in tuple_features.keys():
                for i in range(len(tuple_features[k])):
                    tuple_features[k][i] = tf.squeeze(tuple_features[k][i], 0)

        predicate_atoms2embeddings = {}
        for _, predicate in self.ontology.predicates.items():
            domains = tuple([domain.name for domain in predicate.domains])
            X = tuple_features[domains]
            # print('X', predicate.name, len(X), X[0].shape)
            embeddings = self.embedders[predicate.name](X)
            if self.drop_relational_dim:
                embeddings = tf.expand_dims(embeddings, 0)

            if self.normalize:
                embeddings = tf.math.l2_normalize(embeddings, axis=-1)

            predicate_atoms2embeddings[predicate.name] = embeddings

        atom_embeddings = self.ontology.fol_dictionary_to_linear_tf(
            predicate_atoms2embeddings
        )
        return atom_embeddings

class TransformersBasedFormulaEmbeddingLayer(tf.keras.layers.Layer):
    """Reasoning layer based on Transformer.

    Creates the reasoning layers of the network whose architecture depends on:
    - the number of chained transformers
    - the embedding size
    - the number of atoms of each grounded formula

    Model call:
    reasoning_input -> TransformersBasedFormulaEmbeddingLayer -> reasoning_embedding
    """

    def __init__(
        self,
        o: knowledge.Ontology,
        grounded_formulas: list,
        params: utils.Params,
        layer_name: str = "TransformersBasedFormulaEmbeddingLayer",
    ):
        """Class constructor.

        Args:
            o (`mme.knowledge.Ontology`): dataset ontology
            grounded_formulas (list of `utils.GroundedFormula`): list of grounded formulas
            params (`utils.Params`): model and task parameters, given as default or by command line
            layer_name (`str`, optional): layer's name. Defaults to 'TransformersBasedFormulaEmbeddingLayer'.
        """
        super().__init__()
        assert params.num_transformers > 0

        # list of lists (num_formulas, num_transformers)
        self.transformer_layers = []
        self.grounded_formulas = grounded_formulas

        self.atom_embedding_size = (
            params.atom_embedding_size[-1]
            if type(params.atom_embedding_size) is list
            else params.atom_embedding_size
        )
        self.embedding_sizes = params.transformer_embedding_size

        # overall number of grounded atoms in the ontology
        self.ontology_linear_size = o.linear_size()
        self.params = params
        self.layer_name = layer_name
        self.num_transformers = (
            1 if params.share_network_transformers else self.params.num_transformers
        )

        self.regularizer = (
            tf.keras.regularizers.l2(self.params.transformer_embedder_regularization)
            if self.params.transformer_embedder_regularization > 0.0
            else None
        )

        self.dropout_layer = (
            tf.keras.layers.Dropout(self.params.transformer_input_dropout)
            if self.params.transformer_input_dropout > 0.0
            else None
        )

        # ------------ Architecture building ------------
        # At the end, transformer_layers will be a list of lists of Dense layers.
        # The dimensions are (num_grounded_formulas, num_transformers).
        # So, for each formula there will be a list of num_transformers Dense layers,
        # having units = embedding_size
        for formula_idx, grounded_formula in enumerate(self.grounded_formulas):
            num_atoms = grounded_formula.num_atoms()
            self.transformer_layers.append([])
            embedding_size = (
                self.embedding_sizes[formula_idx]
                if type(self.embedding_sizes) is list
                else self.embedding_sizes
            ) * num_atoms
            for j in range(self.num_transformers):
                transformer_layer = tf.keras.layers.Dense(
                    embedding_size,
                    activation=tf.nn.relu,
                    name="%s_transformer_%d_%d" % (self.layer_name, formula_idx, j),
                    kernel_regularizer=self.regularizer,
                )
                self.transformer_layers[-1].append(transformer_layer)
        # -----------------------------------------------

        if params.attention_formulas:
            attention_shape = [len(self.grounded_formulas), 1, 1]
            self.attention_weights = tf.Variable(
                name="attention_weights",
                shape=attention_shape,
                initial_value=tf.ones(attention_shape) / len(self.grounded_formulas),
                trainable=params.attention_formulas,
            )

    def __merge_clique_embeddings_by_atom__(
        self, atom_embeddings, num_atoms, grounding_indices
    ):
        """Takes the transformer embeddings over the cliques by merging them by average per atom.

        Args:
            atom_embeddings ([type]): [description]
            num_atoms ([type]): [description]
            grounding_indices ([type]): [description]

        Returns:
            [type]: [description]
        """
        num_groundings = grounding_indices.shape[0]

        ones_for_avg = tf.ones(shape=[num_groundings, num_atoms, 1])
        base = tf.zeros([self.ontology_linear_size, atom_embeddings.shape[-1]])
        base_count = tf.zeros([self.ontology_linear_size, 1])

        grounding_indices = tf.expand_dims(grounding_indices, -1)
        aggregated_sum = tf.tensor_scatter_nd_add(
            base, grounding_indices, atom_embeddings
        )
        count = tf.tensor_scatter_nd_add(base_count, grounding_indices, ones_for_avg)
        return tf.math.divide_no_nan(aggregated_sum, count)

    def __embed_atoms_for_formula__(
        self, transformer_idx, atom_embeddings, grounded_formula, transformer_layers
    ):
        """Embed the atoms of the formula by using the current context to predict the transformed assignments.

        This can be a good solution for cases where input embeddings are strong
        and generalization can happen at the pure representation level.

        Given the grounded formula, its parameter `grounding_indices` represents a 2D tensor having a number
        of rows equal to the number of possible groundings for that formula (given the filter) and a number
        of columns equal to the number of atoms in the formula. Each row represents a possible grounding and
        it is composed of `num_atoms` indices, each one representing a possible atom.

        Each index in each row is converted to the related `atom_embedding`, thus obtaining a 3D tensor having
        dimensions (num_groundings, num_atoms, embedding_size). This tensor is then reshaped back to a 2D tensor
        with shape (num_groundings, num_atoms*embedding_size), which can be passed as input to the

        Args:
            transformer_idx (int): index of the transformer, among the chained transformers
            atom_embeddings (tf.tensor): atom embeddings of dimensions (ontology_linear_size, embedding_size)
            grounded_formula (`utils.GroundedFormula`): an instance of class GroundedFormula
            transformer_layers (list of `keras.layers.Dense`): list of Dense layers

        Returns:
            Transformed atom embeddings, depending on the formula
        """

        num_atoms = grounded_formula.formula.num_atoms()
        grounding_indices = grounded_formula.grounding_indices
        # (filtered_groundings, num_atoms)
        by_clique_embeddings = tf.gather(
            indices=grounding_indices, params=atom_embeddings, axis=0
        )
        # (filtered_groundings, num_atoms, embedding_size)
        shape = by_clique_embeddings.shape
        index = min(transformer_idx, self.num_transformers - 1)
        by_clique_embeddings = tf.reshape(by_clique_embeddings, [shape[0], -1])
        by_clique_embeddings = transformer_layers[index](by_clique_embeddings)
        by_clique_embeddings = tf.reshape(
            by_clique_embeddings, [shape[0], shape[1], -1]
        )

        return self.__merge_clique_embeddings_by_atom__(
            by_clique_embeddings, num_atoms, grounding_indices
        )

    def call(self, inputs, **kwargs):
        """Call the layer on the inputs.

        Args:
            inputs (tf.Tensor): atom embeddings of dimensions (1, ontology_linear_size, embedding_size)

        Returns:
            tf.Tensor: reasoning embeddings
        """
        inputs = tf.squeeze(inputs, axis=0)  # drop the relational dimension

        atom_embeddings = self.dropout_layer(inputs) if self.dropout_layer else inputs

        # ------------ Forward pass ------------
        for transformer_idx in range(self.params.num_transformers):
            formula_atom_embeddings = []
            for (grounded_formula, transformer_layers) in zip(
                self.grounded_formulas, self.transformer_layers
            ):
                one_formula_atom_embeddings = self.__embed_atoms_for_formula__(
                    transformer_idx,
                    atom_embeddings,
                    grounded_formula,
                    transformer_layers,
                )
                formula_atom_embeddings.append(one_formula_atom_embeddings)

            # Compute mean over the formulas.
            atom_embeddings = tf.stack(formula_atom_embeddings, 0)
            if self.params.attention_formulas:
                weights = tf.nn.softmax(self.attention_weights)
                atom_embeddings = atom_embeddings * weights
            atom_embeddings = tf.math.reduce_sum(atom_embeddings, axis=0)
        # --------------------------------------

        atom_embeddings = tf.expand_dims(atom_embeddings, 0)
        return atom_embeddings


class MaskedTransformersBasedFormulaEmbeddingLayer(
    TransformersBasedFormulaEmbeddingLayer
):
    def __init__(
        self,
        o,
        grounded_formulas,
        params,
        layer_name="MaskedTransformersBasedFormulaEmbeddingLayer",
    ):
        super(MaskedTransformersBasedFormulaEmbeddingLayer, self).__init__(
            o, grounded_formulas, params, layer_name=layer_name
        )
        assert params.num_transformers > 0

        # list of lists
        # (num_formulas, num_transformers)
        self.transformer_layers = []

        self.grounded_formulas = grounded_formulas
        # The atom embedding size as input.
        self.atom_embedding_size = (
            params.atom_embedding_size[-1]
            if type(params.atom_embedding_size) is list
            else params.atom_embedding_size
        )
        self.embedding_sizes = params.transformer_embedding_size
        # This is the global number of atoms.
        self.ontology_linear_size = o.linear_size()
        self.params = params
        self.layer_name = layer_name

        self.num_transformers = (
            1 if params.share_network_transformers else self.params.num_transformers
        )

        self.regularizer = None
        if self.params.transformer_embedder_regularization > 0.0:
            self.regularizer = tf.keras.regularizers.l2(
                self.params.transformer_embedder_regularization
            )

        self.dropout_layer = None
        if self.params.transformer_input_dropout > 0.0:
            self.dropout_layer = tf.keras.layers.Dropout(
                self.params.transformer_input_dropout
            )

        for i, grounded_formula in enumerate(self.grounded_formulas):
            formula = grounded_formula.formula
            grounding_indices = grounded_formula.grounding_indices
            cardinality = grounding_indices.shape[1]
            self.transformer_layers.append([])
            embedding_size = self.__compute_embedding_size(i, cardinality)
            for j in range(self.num_transformers):
                transformer_layer = tf.keras.layers.Dense(
                    embedding_size,
                    activation=tf.nn.relu,
                    name="%s_transformer_%d_%d" % (self.layer_name, i, j),
                    kernel_regularizer=self.regularizer,
                )
                self.transformer_layers[-1].append(transformer_layer)

        if params.attention_formulas:
            attention_shape = [len(self.grounded_formulas), 1, 1]
            self.attention_weights = tf.Variable(
                name="attention_weights",
                shape=attention_shape,
                initial_value=tf.ones(attention_shape) / len(self.grounded_formulas),
                trainable=params.attention_formulas,
            )

        self.masked_grounding_indices = []
        self.num_atoms = []
        self.num_groundings = []
        for grounded_formula in self.grounded_formulas:
            formula, grounding_indices = (
                grounded_formula.formula,
                grounded_formula.grounding_indices,
            )
            num_atoms = formula.num_atoms()
            self.num_atoms.append(num_atoms)
            num_groundings = grounding_indices.shape[0]
            self.num_groundings.append(num_groundings)

            # num_groundings x 1 x num_atoms
            expanded_indices = tf.expand_dims(grounding_indices, 1)

            # num_groundings x num_atoms x num_atoms
            tiled_grounding_indices = tf.tile(expanded_indices, [1, num_atoms, 1])
            # num_atoms x num_atoms
            mask = tf.logical_not(tf.eye(num_atoms, dtype=tf.bool))

            # 1 x num_atoms x num_atoms
            expanded_mask = tf.expand_dims(mask, 0)
            # num_groundings x num_atoms x num_atoms
            tiled_mask = tf.tile(expanded_mask, [num_groundings, 1, 1])

            # Remove the elements False.
            self.masked_grounding_indices.append(
                tf.reshape(
                    tf.boolean_mask(tiled_grounding_indices, tiled_mask),
                    [num_groundings, num_atoms, num_atoms - 1],
                )
            )

    def __compute_embedding_size(self, i, cardinality):
        embedding_size = (
            self.embedding_sizes[i]
            if type(self.embedding_sizes) is list
            else self.embedding_sizes
        )
        return embedding_size

    def __embed_atoms_for_formula__(
        self, i, j, atom_embeddings, grounded_formula, transformer_layers
    ):
        """
        Embed the atoms of the formula by masking the element to predict the next
        assignment based on the current context. This solution allows to properly
        generalize even when the input representation is poor over test data, as
        only the context will be used to predict the element itself.
        i is the transformer index
        j is the formula index

        [extended_summary]

        Args:
            i ([type]): [description]
            j ([type]): [description]
            atom_embeddings ([type]): [description]
            grounded_formula ([type]): [description]
            transformer_layers ([type]): [description]

        Returns:
            [type]: [description]
        """
        formula, grounding_indices = (
            grounded_formula.formula,
            grounded_formula.grounding_indices,
        )
        num_atoms = formula.num_atoms()
        num_groundings = grounding_indices.shape[0]

        index = min(i, self.num_transformers - 1)

        # masked embeddings will have shape:
        # (num_groundings, num_atoms, num_atoms-1, atom_embedding_size)
        masked_embeddings = tf.gather(
            indices=self.masked_grounding_indices[j], params=atom_embeddings, axis=0
        )
        if i == 0:
            embedding_size = self.atom_embedding_size
        else:
            embedding_size = (
                self.embedding_sizes[i - 1]
                if type(self.embedding_sizes) is list
                else self.embedding_sizes
            )
        new_shape = (
            self.num_groundings[j],
            self.num_atoms[j],
            (self.num_atoms[j] - 1) * embedding_size,
        )
        masked_embeddings = tf.reshape(masked_embeddings, new_shape)

        masked_embeddings = transformer_layers[index](masked_embeddings)
        one_formula_atom_embeddings = self.__merge_clique_embeddings_by_atom__(
            masked_embeddings, num_atoms, grounding_indices
        )
        return one_formula_atom_embeddings

    def __merge_clique_embeddings_by_atom__(
        self, atom_embeddings, num_atoms, grounding_indices
    ):
        """Takes the transformer embeddings over the cliques by merging them by average
        per atom.

        [extended_summary]

        Args:
            atom_embeddings ([type]): [description]
            num_atoms ([type]): [description]
            grounding_indices ([type]): [description]

        Returns:
            [type]: [description]
        """
        num_groundings = grounding_indices.shape[0]

        ones_for_avg = tf.ones(shape=[num_groundings, num_atoms, 1])
        base = tf.zeros([self.ontology_linear_size, atom_embeddings.shape[-1]])
        base_count = tf.zeros([self.ontology_linear_size, 1])

        grounding_indices = tf.expand_dims(grounding_indices, -1)
        aggregated_sum = tf.tensor_scatter_nd_add(
            base, grounding_indices, atom_embeddings
        )
        count = tf.tensor_scatter_nd_add(base_count, grounding_indices, ones_for_avg)
        return tf.math.divide_no_nan(aggregated_sum, count)

    def call(self, inputs, **kwargs):
        """

        Inputs are an embedded hb [1, o.linear_size(), embedding_size]

        [extended_summary]

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """

        # Drop the relational dimension.
        inputs = tf.squeeze(inputs, axis=0)
        if self.dropout_layer:
            atom_embeddings = self.dropout_layer(inputs)
        else:
            atom_embeddings = inputs

        for i in range(self.params.num_transformers):
            formula_atom_embeddings = []
            for j, data in enumerate(
                zip(self.grounded_formulas, self.transformer_layers)
            ):
                (grounded_formula, transformer_layers) = data
                one_formula_atom_embeddings = self.__embed_atoms_for_formula__(
                    i, j, atom_embeddings, grounded_formula, transformer_layers
                )
                formula_atom_embeddings.append(one_formula_atom_embeddings)

            # Compute mean over the formulas.
            atom_embeddings = tf.stack(formula_atom_embeddings, 0)
            if self.params.attention_formulas:
                weights = tf.nn.softmax(self.attention_weights)
                atom_embeddings = atom_embeddings * weights
            atom_embeddings = tf.math.reduce_sum(atom_embeddings, axis=0)

        atom_embeddings = tf.expand_dims(atom_embeddings, 0)
        return atom_embeddings


class TransformersBasedFormulaOutputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, **kwargs):
        return tf.squeeze(self.output_layer(inputs), axis=-1)


class CliquesOutputLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        ontology: knowledge.Ontology,
        grounded_formulas: list,
        params: utils.Params,
    ):
        super().__init__()
        self.grounded_formulas = [
            gf for gf in grounded_formulas if gf.formula.is_hard()
        ]
        self.ontology = ontology
        self.outputs_layers = []
        for _ in self.grounded_formulas:
            if params.formula_hidden_neurons > 0:
                layer = tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(
                            params.formula_hidden_neurons, activation=tf.nn.sigmoid
                        ),
                        tf.keras.layers.Dropout(params.formula_dropout_prob),
                        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
                    ]
                )
            else:
                layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            self.outputs_layers.append(layer)

    def call(self, inputs):
        atoms_embeddings = inputs
        outputs = []
        for i, grounded_formula in enumerate(self.grounded_formulas):
            cliques_atom_embeddings = tf.gather(
                params=atoms_embeddings,
                indices=grounded_formula.grounding_indices,
                axis=-2,
            )
            shape = tf.shape(cliques_atom_embeddings)
            # concatenation of the atom embeddings per clique
            formula_cliques_embedding = tf.reshape(
                cliques_atom_embeddings,
                [shape[0], shape[1], shape[2] * shape[3]],
            )
            outputs.append(
                tf.squeeze(self.outputs_layers[i](formula_cliques_embedding), -1)
            )
        return tf.concat(outputs, axis=-1)


class SemanticOutputLayer(tf.keras.layers.Layer):
    """Takes the atom embeddings outputs and select the ones for the cliques.
    Then, applies the logic on those real value outputs and imposes the supervision."""

    def __init__(
        self,
        ontology: knowledge.Ontology,
        grounded_formulas: list,
        # params: utils.Params,
    ):
        super().__init__()
        self.grounded_formulas = [
            gf for gf in grounded_formulas if gf.formula.is_hard()
        ]
        self.ontology = ontology

    def call(self, inputs: list):
        """Applies the semantic of the logic formulas on the atom embeddings output (real
        values).

        [extended_summary]

        Args:
            inputs (tf.Tensor):
                Atom embeddings output predictions (real values between 0 and 1) of shape
                (1, ontology.linear_size()).

        Returns:
            outputs (tf.Tensor):
                Clique predictions (real values between 0 and 1) of shape
                (1, num_cliques_formula1+num_cliques_formula2+...+num_cliques_formulaN).
        """
        outputs = []
        for _, grounded_formula in enumerate(self.grounded_formulas):
            cliques_atom_embeddings = tf.gather(
                params=inputs,
                indices=grounded_formula.grounding_indices,
                axis=-1,
            )
            # TODO(filippo): how to perform logic operations between real numbers?
            # Possible solutions:
            #   1)
            #       - AND: Real x Real -> Boolean
            #              AND(x1, x2) = True if min(sigm(x1), sigm(x2)) > 0.5 else False
            #       - OR:  Real x Real -> Boolean
            #              OR(x1, x2) = True if max(sigm(x1), sigm(x2)) > 0.5 else False
            #       - NOT: Real -> Real
            #              NOT(x1) = 1 - sigm(x1)
            #       - IMPLICATION: Real x Real -> Boolean
            #              IMPLIES(x1, x2) = OR(NOT(sigm(x1)), sigm(x2))
            #   2) Lukasiewicz Logic or Product Logic

            outputs.append(
                grounded_formula.formula.evaluate(
                    cliques_atom_embeddings, logic.LukasiewiczLogic
                )
            )

        return tf.expand_dims(tf.concat(outputs, axis=-1), axis=0)


##############################FGNN
class FGNNBasedFormulaEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, o, grounded_formulas, params,
                 layer_name='FGNNBasedFormulaEmbeddingLayer'):
        super(FGNNBasedFormulaEmbeddingLayer,self).__init__()
        assert params.num_transformers > 0

        # list of lists
        # (num_formulas, num_transformers)
        self.transformer_layers = []

        self.grounded_formulas = grounded_formulas
        # The atom embedding size as input.
        self.atom_embedding_size = (params.atom_embedding_size[-1]
                                    if type(params.atom_embedding_size) is list
                                    else params.atom_embedding_size)
        self.embedding_sizes = params.transformer_embedding_size
        # This is the global number of atoms.
        self.ontology_linear_size = o.linear_size()
        self.params = params
        self.layer_name = layer_name

        self.num_transformers = (1
                                 if params.share_network_transformers
                                 else self.params.num_transformers)

        self.regularizer = None
        if self.params.transformer_embedder_regularization > 0.0:
            self.regularizer = tf.keras.regularizers.l2(
                self.params.transformer_embedder_regularization)

        self.dropout_layer = None
        if self.params.transformer_input_dropout > 0.0:
            self.dropout_layer = tf.keras.layers.Dropout(
                self.params.transformer_input_dropout)


        gc_size = self.atom_embedding_size
        self.first_gcs = []
        for i,grounded_formula in enumerate(self.grounded_formulas):
            grounding_indices = grounded_formula.grounding_indices
            formula = grounded_formula.formula
            self.transformer_layers.append([])
            # 1)
            # self.first_gcs.append(tf.Variable(initial_value=tf.zeros([len(grounding_indices), gc_size])))

            # 2)
            self.first_gcs.append(tf.random.normal(shape=[len(grounding_indices), gc_size]))

            # 3)
            # evaluation = build_one_formula_targets(grounded_formula, params.y_for_inputs)
            # evaluation = tf.reshape(evaluation, [-1,1])
            # self.first_gcs.append(evaluation)

            # 4)
            # groundings = build_one_formula_targets(grounded_formula, params.y_for_inputs)
            # groundings = tf.reshape(groundings, [grounding_indices.shape[0], -1])
            # self.first_gcs.append(groundings)


            for j in range(self.num_transformers):
                embedding_size = (self.embedding_sizes[j]
                                  if type(self.embedding_sizes) is list
                                  else self.embedding_sizes)
                M = tf.keras.layers.Dense(
                    embedding_size, activation=tf.nn.relu,
                    name='%s_FGNN_M_%d_%d'%(self.layer_name, i, j),
                    kernel_regularizer=self.regularizer)
                U = tf.keras.layers.Dense(
                    embedding_size, activation=tf.nn.relu,
                    name='%s_FGNN_U_%d_%d'%(self.layer_name, i, j),
                    kernel_regularizer=self.regularizer)
                self.transformer_layers[-1].append((M,U))

        if  params.attention_formulas:
            attention_shape = [len(self.grounded_formulas), 1, 1]
            self.attention_weights = tf.Variable(name="attention_weights", shape = attention_shape, initial_value=tf.ones(attention_shape) / len(self.grounded_formulas), trainable= params.attention_formulas)



    def __merge_clique_embeddings_by_atom__(self,
                                            atom_embeddings, num_atoms,
                                            grounding_indices):
        num_groundings = grounding_indices.shape[0]

        ones_for_avg = tf.ones(shape=[num_groundings, num_atoms, 1])
        base = tf.zeros([self.ontology_linear_size, atom_embeddings.shape[-1]])
        base_count = tf.zeros([self.ontology_linear_size, 1])

        grounding_indices = tf.expand_dims(grounding_indices, -1)
        aggregated_sum = tf.tensor_scatter_nd_add(
            base, grounding_indices, atom_embeddings)
        count = tf.tensor_scatter_nd_add(base_count, grounding_indices,
                                         ones_for_avg)
        return tf.math.divide_no_nan(aggregated_sum, count)

    # Embed the atoms of the formula by using the current context to predict the
    # transformed assignments. This can be a good solution for cases where input
    # embeddings are strong and generalization can happen at the pure
    # representation level.

    def call(self, inputs, **kwargs):
        # Inputs are an embedded hb [1, o.linear_size(), embedding_size]
        # Drop the relational dimension.
        inputs = tf.squeeze(inputs, axis=0)
        if self.dropout_layer:
            atom_embeddings = self.dropout_layer(inputs)
        else:
            atom_embeddings = inputs
        last_gc = [None for _ in self.grounded_formulas]
        for i in range(self.params.num_transformers):
            inputs_networks = []
            embedding_size = (self.embedding_sizes[i]
                              if type(self.embedding_sizes) is list
                              else self.embedding_sizes)
            new_atom_embeddings = tf.zeros(shape = [atom_embeddings.shape[0], embedding_size]) # #TODO: this is safe only for relu computed embeddings
            for j, grounded_formula in enumerate(self.grounded_formulas):
                grounding_indices = grounded_formula.grounding_indices

                gc_fi = tf.gather(indices=grounding_indices,
                                  params=atom_embeddings,
                                  axis=0)
                gcl = self.first_gcs[j] if i==0 else last_gc[j]
                tiled_gc = tf.tile(tf.expand_dims(gcl, -2), [1, grounding_indices.shape[1], 1])
                gc_fi = tf.concat((gc_fi,tiled_gc),axis=-1)
                inputs_networks.append(gc_fi)

                (Mc,Mn) = self.transformer_layers[j][i]

                if i < self.params.num_transformers - 1:
                    gc = Mc(gc_fi)
                    gc = tf.reduce_max(gc, axis=-2)
                    last_gc[j] = gc

                fi = Mn(gc_fi)
                grounding_indices = tf.expand_dims(grounding_indices, -1)
                new_atom_embeddings = tf.tensor_scatter_nd_max(new_atom_embeddings, tf.cast(grounding_indices, tf.int32), fi)
            atom_embeddings = new_atom_embeddings
        atom_embeddings = tf.expand_dims(atom_embeddings, 0)
        return atom_embeddings
