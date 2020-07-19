from ..utils import common_functions as c_f

class BaseEnsemble:
    def __init__(self, normalize_embeddings=True, use_trunk_output=False):
        self.normalize_embeddings = normalize_embeddings
        self.use_trunk_output = use_trunk_output

    def get_trunk_and_embedder(self):
        raise NotImplementedError

    def get_eval_record_name_dict(self, hooks, tester, split_names):
        base_output = c_f.get_eval_record_name_dict(hooks, tester, split_names=split_names)
        return {k:"{}_{}".format(self.__class__.__name__, v) for k,v in base_output.items()}