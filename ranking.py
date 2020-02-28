import numpy as np

from typing import List, Tuple, Dict, Callable, Optional



class Ranking():
    
    def __init__(self,
                 test_triples: List[Tuple[str, str, str]],
                 all_triples: List[Tuple[str, str, str]],
                 entity_indices: np.ndarray):

    
        self.test_triples = test_triples
        self.all_triples = all_triples
    
        entities = entity_indices.tolist()
    
        self.corruptions = {}
        for s_idx, p_idx, o_idx in test_triples:
            corrupted_subject = [(entity, p_idx, o_idx) for entity in entities if (entity, p_idx, o_idx) not in all_triples or entity == s_idx]
            corrupted_object = [(s_idx, p_idx, entity) for entity in entities if (s_idx, p_idx, entity) not in all_triples or entity == o_idx]
    
            index_l = corrupted_subject.index((s_idx, p_idx, o_idx))
            index_r = corrupted_object.index((s_idx, p_idx, o_idx))
    
            nb_corrupted_l = len(corrupted_subject)
            # nb_corrupted_r = len(corrupted_object)
    
            corrupted = corrupted_subject + corrupted_object

            self.corruptions[s_idx, p_idx, o_idx] = (index_l, index_r, corrupted, nb_corrupted_l)

        self.max = {"hits@1":0.0}
        
    def evaluation(self,scoring_function: Callable):

        hits = dict()
        hits_at = [1, 3, 5, 10]

        for hits_at_value in hits_at:
            hits[hits_at_value] = 0.0

        def hits_at_n(n_, rank):
            if rank <= n_:
                hits[n_] = hits.get(n_, 0) + 1

        counter = 0
        mrr = 0.0

        for s_idx, p_idx, o_idx in self.test_triples:

            (index_l, index_r, corrupted, nb_corrupted_l) = self.corruptions[s_idx, p_idx, o_idx]


            scores_lst = scoring_function(corrupted)

            scores_l = scores_lst[:nb_corrupted_l]
            scores_r = scores_lst[nb_corrupted_l:]
    
            rank_l = 1 + np.argsort(np.argsort(- np.array(scores_l)))[index_l]
            counter += 1
    
            for n in hits_at:
                hits_at_n(n, rank_l)
    
            mrr += 1.0 / rank_l
    
            rank_r = 1 + np.argsort(np.argsort(- np.array(scores_r)))[index_r]
            counter += 1

            for n in hits_at:
                hits_at_n(n, rank_r)
    
            mrr += 1.0 / rank_r
    
        counter = float(counter)
    
        mrr /= counter
    
        for n in hits_at:
            hits[n] /= counter
    
        metrics = dict()
        metrics['MRR'] = mrr
        for n in hits_at:
            metrics['hits@{}'.format(n)] = hits[n]

        updated = False
        if hits[1]>self.max["hits@1"]:
            self.max = metrics
            updated = True

    
        return (mrr, hits[1], hits[3], hits[5], hits[10]), updated