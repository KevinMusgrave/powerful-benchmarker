from ..utils import common_functions as c_f
import torch
from .base_ensemble import BaseEnsemble
from .. import architectures

class ConcatenateEmbeddings(BaseEnsemble):
    def create_ensemble_model(self, list_of_trunks, list_of_embedders):
        if isinstance(self.embedder_input_sizes[0], list):
            self.embedder_input_sizes = [np.sum(x) for x in self.embedder_input_sizes]
        normalize_embeddings_func = lambda x: torch.nn.functional.normalize(x, p=2, dim=1)
        embedder_operation_before_concat = normalize_embeddings_func if self.normalize_embeddings else None
        trunk_operation_before_concat = normalize_embeddings_func if self.use_trunk_output else None

        trunk = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_trunks, operation_before_concat=trunk_operation_before_concat))
        embedder = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_embedders, self.embedder_input_sizes, embedder_operation_before_concat))
        return trunk, embedder