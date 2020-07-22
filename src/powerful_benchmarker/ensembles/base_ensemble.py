from ..utils import common_functions as c_f

class BaseEnsemble:
    def __init__(self, normalize_embeddings=True, use_trunk_output=False):
        self.normalize_embeddings = normalize_embeddings
        self.use_trunk_output = use_trunk_output

    def get_list_of_models(self, model_factory, model_args, model_name, factory_kwargs, split_folders, device):
        list_of_trunks, list_of_embedders = [], []
        for model_folder in split_folders:
            trunk_model, embedder_model = c_f.load_model_for_eval(model_factory, model_args, model_name, factory_kwargs, model_folder, device)
            list_of_trunks.append(trunk_model.module)
            list_of_embedders.append(embedder_model.module)
        self.embedder_input_sizes = [model_factory.base_model_output_size] * len(list_of_trunks)
        return list_of_trunks, list_of_embedders

    def get_trunk_and_embedder(self):
        raise NotImplementedError

    def get_eval_record_name_dict(self, hooks, tester, split_names):
        base_output = c_f.get_eval_record_name_dict(hooks, tester, split_names=split_names)
        return {k:"{}_{}".format(self.__class__.__name__, v) for k,v in base_output.items()}