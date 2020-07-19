from ..utils import common_functions as c_f
import torch
from .base_ensemble import BaseEnsemble
from .. import architectures

class ConcatenateEmbeddings(BaseEnsemble):
    def get_trunk_and_embedder(self, model_factory, model_args, model_name, split_folders, device):
        list_of_trunks, list_of_embedders = [], []
        for model_folder in split_folders:
            trunk_model, embedder_model = c_f.load_model_for_eval(model_factory, model_args, model_name, model_folder, device)
            list_of_trunks.append(trunk_model.module)
            list_of_embedders.append(embedder_model.module)
        embedder_input_sizes = [model_factory.base_model_output_size] * len(list_of_trunks)
        if isinstance(embedder_input_sizes[0], list):
            embedder_input_sizes = [np.sum(x) for x in embedder_input_sizes]
        normalize_embeddings_func = lambda x: torch.nn.functional.normalize(x, p=2, dim=1)
        embedder_operation_before_concat = normalize_embeddings_func if self.normalize_embeddings else None
        trunk_operation_before_concat = normalize_embeddings_func if self.use_trunk_output else None

        trunk = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_trunks, operation_before_concat=trunk_operation_before_concat))
        embedder = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_embedders, embedder_input_sizes, embedder_operation_before_concat))
        return trunk, embedder