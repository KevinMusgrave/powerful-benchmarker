class BaseEnsemble:
    def __init__(self, normalize_embeddings=True, use_trunk_output=False):
        self.normalize_embeddings = normalize_embeddings
        self.use_trunk_output = use_trunk_output

    def get_trunk_and_embedder(self):
        raise NotImplementedError


