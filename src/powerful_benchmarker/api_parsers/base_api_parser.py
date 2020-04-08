import sys
import copy
from ..utils import common_functions as c_f, split_manager
from pytorch_metric_learning import losses
import pytorch_metric_learning.utils.logging_presets as logging_presets
import pytorch_metric_learning.utils.common_functions as pml_cf
import pytorch_metric_learning.utils.calculate_accuracies as pml_ca
from torch.utils.tensorboard import SummaryWriter
import torch.nn
import torch
import os
import pathlib
import shutil
import logging
import numpy as np
from scipy import stats as scipy_stats
from collections import defaultdict
from .. import architectures


class BaseAPIParser:
    def __init__(self, args, pytorch_getter, global_db_path=None):
        pml_cf.NUMPY_RANDOM = np.random.RandomState()
        logging.info("NUMPY_RANDOM = %s"%pml_cf.NUMPY_RANDOM)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.pytorch_getter = pytorch_getter
        self.experiment_folder = args.experiment_folder
        self.global_db_path = global_db_path

        _model_folder = os.path.join("%s", "%s", "saved_models")
        _csv_folder = os.path.join("%s", "%s", "saved_csvs")
        _tensorboard_folder = os.path.join("%s", "%s", "tensorboard_logs")
        self.sub_experiment_dirs = [
            _model_folder,
            _csv_folder,
            _tensorboard_folder
        ]

        self.trainer, self.tester_obj = None, None

    def run(self):
        if self.beginning_of_training():
            self.make_dir()
        self.set_split_manager()
        self.save_config_files()
        self.set_num_epochs_dict()
        self.make_sub_experiment_dirs()
        self.set_meta_record_keeper()
        if self.args.evaluate and self.args.meta_testing_method:
            self.meta_eval()
        else:
            try:
                self.run_for_each_split_scheme()
            except ValueError as value_exception:
                error_string = str(value_exception)
                if "NaN" in error_string:
                    logging.error(error_string)
                    mean, sem = 0, 0
                    self.delete_old_objects()
                    return mean, sem if hasattr(self, "meta_record_keeper") else mean
                else:
                    raise ValueError
            self.record_meta_logs()
        self.flush_tensorboard()
        self.delete_old_objects()
        if self.is_training():
            return self.return_val_accuracy_and_standard_error()

    def run_for_each_split_scheme(self):
        for self.curr_split_count, split_scheme_name in enumerate(self.split_manager.split_scheme_names):
            num_epochs = self.num_epochs[split_scheme_name]
            self.split_manager.set_curr_split_scheme(split_scheme_name)
            self.set_curr_folders()
            self.set_models_optimizers_losses()
            if self.args.evaluate:
                self.eval()
            elif self.should_train(num_epochs, split_scheme_name):
                self.train(num_epochs)
            self.update_meta_record_keeper(split_scheme_name)
            self.delete_old_objects()

    def delete_old_objects(self):
        for attr_name in ["models", "loss_funcs", "mining_funcs", "optimizers", "lr_schedulers", "gradient_clippers"]:
            try:
                delattr(self, attr_name)
                logging.info("Deleted self.%s"%attr_name)
            except AttributeError:
                pass
        try:
            del self.trainer.dataloader
            del self.trainer.dataloader_iter
            logging.info("Deleted trainer dataloader")
        except AttributeError:
            pass

    def is_training(self):
        return not self.args.evaluate

    def beginning_of_training(self):
        return (not self.args.resume_training) and (not self.args.evaluate)

    def make_dir(self):
        root = pathlib.Path(self.experiment_folder)
        non_empty_dirs = {str(p.parent) for p in root.rglob('*') if p.is_file()}
        non_empty_dirs.discard(self.experiment_folder)
        if len(non_empty_dirs) > 0:
            logging.info("Experiment folder already taken!")
            sys.exit()
        c_f.makedir_if_not_there(self.experiment_folder)

    def make_sub_experiment_dirs(self):
        for s in self.sub_experiment_dirs:
            for r in self.split_manager.split_scheme_names:
                c_f.makedir_if_not_there(s % (self.experiment_folder, r))

    def set_curr_folders(self):
        self.model_folder, self.csv_folder, self.tensorboard_folder = self.get_sub_experiment_dir_paths()[self.split_manager.curr_split_scheme_name]

    def get_sub_experiment_dir_paths(self):
        sub_experiment_dir_paths = {}
        for k in self.split_manager.split_scheme_names:
            sub_experiment_dir_paths[k] = [s % (self.experiment_folder, k) for s in self.sub_experiment_dirs]
        return sub_experiment_dir_paths

    def set_num_epochs_dict(self):
        if isinstance(self.args.num_epochs_train, int):
            self.num_epochs = {k: self.args.num_epochs_train for k in self.split_manager.split_scheme_names}
        else:
            self.num_epochs = self.args.num_epochs_train

    def save_config_files(self):
        self.latest_sub_experiment_epochs = c_f.latest_sub_experiment_epochs(self.get_sub_experiment_dir_paths())
        latest_epochs = list(self.latest_sub_experiment_epochs.values())
        if self.is_training():
            c_f.save_config_files(self.args.place_to_save_configs, self.args.dict_of_yamls, self.args.resume_training, self.args.reproduce_results, latest_epochs)
        delattr(self.args, "dict_of_yamls")
        delattr(self.args, "place_to_save_configs")

    def set_optimizers(self):
        self.optimizers, self.lr_schedulers, self.gradient_clippers = {}, {}, {}
        for k, v in self.args.optimizers.items():
            basename = k.replace("_optimizer", '')
            param_source = None
            for possible_params in [self.models, self.loss_funcs]:
                if basename in possible_params:
                    param_source = possible_params[basename]
                    break
            o, s, g = self.pytorch_getter.get_optimizer(param_source, yaml_dict=v)
            logging.info("%s\n%s" % (k, o))
            if o is not None: self.optimizers[k] = o
            if s is not None: self.lr_schedulers[basename + "_scheduler"] = s
            if g is not None: self.gradient_clippers[basename + "_grad_clipper"] = g

    def set_split_manager(self):
        chosen_dataset = self.pytorch_getter.get("dataset", yaml_dict=self.args.dataset, additional_params={"dataset_root":self.args.dataset_root})
        self.split_manager = split_manager.SplitManager(dataset=chosen_dataset, 
                                                        test_size=self.args.test_size,
                                                        test_start_idx=self.args.test_start_idx, 
                                                        num_training_partitions=self.args.num_training_partitions,
                                                        num_training_sets=self.args.num_training_sets,
                                                        special_split_scheme_name=self.args.special_split_scheme_name,
                                                        hierarchy_level=self.args.label_hierarchy_level)
        if not self.args.splits_to_eval: self.args.splits_to_eval = [x for x in self.split_manager.split_names if x!="test"]

    def get_transforms(self):
        try:
            trunk = self.models["trunk"]
            if isinstance(trunk, torch.nn.DataParallel):
                trunk = trunk.module
            model_transform_properties = {k:getattr(trunk, k) for k in ["mean", "std", "input_space", "input_range"]}
        except (KeyError, AttributeError):
            model_transform_properties = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        transforms = {"train": None, "eval": None}
        for k, v in self.args.transforms.items():
            transforms[k] = self.pytorch_getter.get_composed_img_transform(v, **model_transform_properties)
        return transforms

    def set_transforms(self):
        logging.info("Setting dataset so that transform can be set")
        self.split_manager.set_curr_split("train", is_training=True)
        transforms = self.get_transforms()
        self.split_manager.set_transforms(transforms["train"], transforms["eval"])

    def get_embedder_model(self, model_type, input_size=None, output_size=None):
        model, model_args = self.pytorch_getter.get("model", yaml_dict=model_type, return_uninitialized=True)
        model_args = copy.deepcopy(model_args)
        if model == architectures.misc_models.MLP:
            if input_size:
                model_args["layer_sizes"].insert(0, input_size)
            if output_size:
                model_args["layer_sizes"].append(output_size)
        model = model(**model_args)
        logging.info("EMBEDDER MODEL %s"%model)
        return model

    def get_trunk_model(self, model_type):
        model = self.pytorch_getter.get("model", yaml_dict=model_type)
        self.base_model_output_size = c_f.get_last_linear(model).in_features
        c_f.set_last_linear(model, architectures.misc_models.Identity())
        return model

    def get_mining_function(self, miner_type):
        if miner_type:
            miner, miner_params = self.pytorch_getter.get("miner", yaml_dict=miner_type, return_uninitialized=True)
            miner_params = copy.deepcopy(miner_params)
            if "loss" in miner_params: miner_params["loss"] = self.get_loss_function(miner_params["loss"])
            if "miner" in miner_params: miner_params["miner"] = self.get_mining_function(miner_params["miner"])
            return miner(**miner_params)
        return None

    def set_sampler(self):
        if self.args.sampler in [None, {}]:
            self.sampler = None
        else:
            self.sampler = self.pytorch_getter.get("sampler", yaml_dict=self.args.sampler, additional_params={"labels":self.split_manager.labels})

    def get_loss_function(self, loss_type):
        loss, loss_params = self.pytorch_getter.get("loss", yaml_dict=loss_type, return_uninitialized=True)
        loss_params = copy.deepcopy(loss_params)
        if loss == losses.MultipleLosses:
            loss_funcs = [self.get_loss_function({k:v}) for k,v in loss_params["losses"].items()]
            return loss(loss_funcs) 
        if loss == losses.CrossBatchMemory:
            if "loss" in loss_params: loss_params["loss"] = self.get_loss_function(loss_params["loss"])
            if "miner" in loss_params: loss_params["miner"] = self.get_mining_function(loss_params["miner"])
        if c_f.check_init_arguments(loss, "num_classes"):
            loss_params["num_classes"] = self.split_manager.get_num_labels()
            logging.info("Passing %d as num_classes to the loss function"%loss_params["num_classes"])
        if "regularizer" in loss_params:
            loss_params["regularizer"] = self.pytorch_getter.get("regularizer", yaml_dict=loss_params[k])
        if "num_class_per_param" in loss_params and loss_params["num_class_per_param"]:
            loss_params["num_class_per_param"] = self.split_manager.get_num_labels()
            logging.info("Passing %d as num_class_per_param to the loss function"%loss_params["num_class_per_param"])

        return loss(**loss_params)        

    def set_loss_function(self):
        self.loss_funcs = {}
        for k, v in self.args.loss_funcs.items():
            self.loss_funcs[k] = self.get_loss_function(v)

    def set_mining_function(self):
        self.mining_funcs = {}
        for k, v in self.args.mining_funcs.items():
            self.mining_funcs[k] = self.get_mining_function(v)

    def model_getter_dict(self):
        return {"trunk": lambda model_type: self.get_trunk_model(model_type),
                "embedder": lambda model_type: self.get_embedder_model(model_type, input_size=self.base_model_output_size)}

    def set_model(self):
        self.models = {}
        for k, v in self.model_getter_dict().items():
            self.models[k] = v(self.args.models[k])

    def load_model_for_eval(self, suffix):
        untrained = suffix == "-1"
        trunk_model = self.get_trunk_model(self.args.models["trunk"])
        if not untrained:
            embedder_model = self.get_embedder_model(self.args.models["embedder"], self.base_model_output_size)
            pml_cf.load_dict_of_models(
                {"trunk": trunk_model, "embedder": embedder_model},
                suffix,
                self.model_folder,
                self.device
            )
        else:
            embedder_model = architectures.misc_models.Identity()
        return torch.nn.DataParallel(trunk_model), torch.nn.DataParallel(embedder_model)

    def get_splits_exclusion_list(self, splits_to_eval):
        if set(splits_to_eval) in [{"train"}, {"val"}, {"train", "val"}]:
            return ["test"]
        return []

    def eval_assertions(self, dataset_dict):
        for k, v in dataset_dict.items():
            dataset = self.split_manager.curr_split_scheme[k]
            assert v is dataset
            assert v.dataset.transform is self.split_manager.eval_transform

    def eval_model(self, epoch, suffix, splits_to_eval=None, load_model=False, skip_eval_if_already_done=True):
        logging.info("Launching evaluation for model %s"%suffix)
        if load_model:
            trunk_model, embedder_model = self.load_model_for_eval(suffix=suffix)
        else:
            trunk_model, embedder_model = self.models["trunk"], self.models["embedder"]
        trunk_model, embedder_model = trunk_model.to(self.device), embedder_model.to(self.device)
        splits_to_exclude = self.get_splits_exclusion_list(splits_to_eval)
        dataset_dict = self.split_manager.get_dataset_dict(inclusion_list=splits_to_eval, exclusion_list=splits_to_exclude, is_training=False)
        self.eval_assertions(dataset_dict)
        return self.hooks.run_tester_separately(self.tester_obj, dataset_dict, epoch, trunk_model, embedder_model, 
                                        splits_to_eval=splits_to_eval, collate_fn=None, skip_eval_if_already_done=skip_eval_if_already_done)

    def flush_tensorboard(self):
        for keeper in ["record_keeper", "meta_record_keeper"]:
            k = getattr(self, keeper, None)
            if k:
                k.tensorboard_writer.flush()
                k.tensorboard_writer.close()

    def set_record_keeper(self):
        is_new_experiment = self.beginning_of_training() and self.curr_split_count == 0
        self.record_keeper, _, _ = logging_presets.get_record_keeper(self.csv_folder, self.tensorboard_folder, self.global_db_path, self.args.experiment_name, is_new_experiment)

    def set_meta_record_keeper(self):
        is_new_experiment = self.beginning_of_training()
        if len(self.split_manager.split_scheme_names) > 1:
            _, csv_folder, tensorboard_folder = [s % (self.experiment_folder, "meta_logs") for s in self.sub_experiment_dirs]
            self.meta_record_keeper, _, _ = logging_presets.get_record_keeper(csv_folder, tensorboard_folder,  self.global_db_path, self.args.experiment_name, is_new_experiment)
            self.meta_accuracies = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def update_meta_record_keeper(self, split_scheme_name):
        if hasattr(self, "meta_accuracies"):
            for split in self.args.splits_to_eval:
                best_split_accuracies, _ = self.hooks.get_accuracies_of_best_epoch(self.tester_obj, split, ignore_epoch=(-1, 0))
                untrained_accuracies = self.hooks.get_accuracies_of_epoch(self.tester_obj, split, -1)
                for accuracies, is_trained in [(untrained_accuracies, 0), (best_split_accuracies, 1)]:
                    if len(accuracies) > 0:
                        accuracy_keys = [k for k in accuracies[0].keys() if any(acc in k for acc in pml_ca.METRICS)]
                        for k in accuracy_keys:
                            self.meta_accuracies[split][is_trained][k][split_scheme_name] = accuracies[0][k]

    def record_meta_logs(self):
        if hasattr(self, "meta_accuracies") and len(self.meta_accuracies) > 0:
            for split in self.args.splits_to_eval:
                group_name = self.get_eval_record_name_dict("meta")[split]
                len_of_existing_records = c_f.try_getting_db_count(self.meta_record_keeper, group_name)
                for i, is_trained in enumerate([0, 1]):
                    curr_dict = self.meta_accuracies[split][is_trained]
                    if len(curr_dict) > 0:
                        averages = {k: np.mean(list(v.values())) for k, v in curr_dict.items()}
                        standard_errors = {"SEM_%s"%k: scipy_stats.sem(list(v.values())) for k, v in curr_dict.items()}
                        global_iteration = len_of_existing_records+i+1
                        self.meta_record_keeper.update_records(averages, global_iteration=global_iteration, input_group_name_for_non_objects=group_name)
                        self.meta_record_keeper.update_records(standard_errors, global_iteration=global_iteration, input_group_name_for_non_objects=group_name)
                        self.meta_record_keeper.update_records({"is_trained": is_trained}, global_iteration=global_iteration, input_group_name_for_non_objects=group_name)
                        self.meta_record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=global_iteration, input_group_name_for_non_objects=group_name)
            self.meta_record_keeper.save_records()

    def return_val_accuracy_and_standard_error(self):
        if hasattr(self, "meta_record_keeper"):
            group_name = self.get_eval_record_name_dict("meta")["val"]
            def get_average_best_and_sem(key):
                is_trained = 1
                sem_key = "SEM_%s"%key
                query = "SELECT {0}, {1} FROM {2} WHERE is_trained=? AND id=(SELECT MAX(id) FROM {2})".format(key, sem_key, group_name)
                return self.meta_record_keeper.query(query, values=(is_trained,), use_global_db=False), key, sem_key
            q, key, sem_key = self.hooks.try_primary_metric(self.tester_obj, get_average_best_and_sem)
            return q[0][key], q[0][sem_key]
        _, best_accuracy = self.hooks.get_best_epoch_and_accuracy(self.tester_obj, "val", ignore_epoch=(-1,0))
        return best_accuracy


    def maybe_load_models_and_records(self):
        return self.hooks.load_latest_saved_models(self.trainer, self.model_folder, self.device)

    def set_models_optimizers_losses(self):
        self.set_model()
        self.set_transforms()
        self.set_sampler()
        self.set_loss_function()
        self.set_mining_function()
        self.set_optimizers()
        self.set_record_keeper()
        self.hooks = logging_presets.HookContainer(self.record_keeper, primary_metric=self.args.eval_primary_metric, validation_split_name="val")
        self.tester_obj = self.pytorch_getter.get("tester", self.args.testing_method, self.get_tester_kwargs())
        self.trainer = self.pytorch_getter.get("trainer", self.args.training_method, self.get_trainer_kwargs())
        self.epoch = self.maybe_load_models_and_records()
        self.set_dataparallel()
        self.set_devices()

    def get_tester_kwargs(self):
        return {
            "reference_set": self.args.eval_reference_set,
            "normalize_embeddings": self.args.eval_normalize_embeddings,
            "use_trunk_output": self.args.eval_use_trunk_output,
            "batch_size": self.args.eval_batch_size,
            "data_device": self.device,
            "dataloader_num_workers": self.args.eval_dataloader_num_workers,
            "pca": self.args.eval_pca,
            "size_of_tsne": self.args.eval_size_of_tsne,
            "data_and_label_getter": lambda data: (data["data"], data["label"]),
            "label_hierarchy_level": self.args.label_hierarchy_level,
            "end_of_testing_hook": self.hooks.end_of_testing_hook 
        }

    def get_end_of_epoch_hook(self):
        logging.info("Creating end_of_epoch_hook kwargs")
        splits_to_exclude = self.get_splits_exclusion_list(self.args.splits_to_eval)
        dataset_dict = self.split_manager.get_dataset_dict(inclusion_list=self.args.splits_to_eval, exclusion_list=splits_to_exclude, is_training=False)
        helper_hook = self.hooks.end_of_epoch_hook(tester=self.tester_obj,
                                                    dataset_dict=dataset_dict,
                                                    model_folder=self.model_folder,
                                                    test_interval=self.args.save_interval,
                                                    patience=self.args.patience,
                                                    test_collate_fn=None)
        def end_of_epoch_hook(trainer):
            torch.cuda.empty_cache()
            for k in dataset_dict.keys(): self.split_manager.set_curr_split(k, is_training=False)
            self.eval_assertions(dataset_dict)
            continue_training = helper_hook(trainer)
            for k in dataset_dict.keys(): self.split_manager.set_curr_split(k, is_training=True)
            self.training_assertions(trainer)
            return continue_training

        return end_of_epoch_hook

    def get_trainer_kwargs(self):
        return {
            "models": self.models,
            "optimizers": self.optimizers,
            "batch_size": self.args.batch_size,
            "sampler": self.sampler,
            "collate_fn": None,
            "loss_funcs": self.loss_funcs,
            "mining_funcs": self.mining_funcs,
            "dataset": self.split_manager.dataset,
            "data_device": self.device,
            "iterations_per_epoch": self.args.iterations_per_epoch,
            "lr_schedulers": self.lr_schedulers,
            "gradient_clippers": self.gradient_clippers,
            "freeze_trunk_batchnorm": self.args.freeze_batchnorm,
            "label_hierarchy_level": self.args.label_hierarchy_level,
            "dataloader_num_workers": self.args.dataloader_num_workers,
            "loss_weights": getattr(self.args, "loss_weights", None),
            "data_and_label_getter": lambda data: (data["data"], data["label"]),
            "dataset_labels": self.split_manager.labels,
            "set_min_label_to_zero": True,
            "end_of_iteration_hook": self.hooks.end_of_iteration_hook,
            "end_of_epoch_hook": self.get_end_of_epoch_hook()
        }

    def set_dataparallel(self):
        for k, v in self.models.items():
            self.models[k] = torch.nn.DataParallel(v)

    def set_devices(self):
        for obj_dict in [self.models, self.loss_funcs, self.mining_funcs]:
            for v in obj_dict.values():
                v.to(self.device)
        for v in self.optimizers.values():
            c_f.move_optimizer_to_gpu(v, self.device)

    def should_train(self, num_epochs, split_scheme_name):
        best_epoch, _ = self.hooks.get_best_epoch_and_accuracy(self.tester_obj, "val", ignore_epoch=(-1,))
        return self.hooks.patience_remaining(self.epoch, best_epoch, self.args.patience) and self.latest_sub_experiment_epochs[split_scheme_name] < num_epochs

    def training_assertions(self, trainer):
        dataset = self.split_manager.curr_split_scheme["train"]
        assert trainer.dataset is dataset
        assert trainer.dataset.dataset.transform is self.split_manager.train_transform
        assert np.array_equal(trainer.dataset_labels, self.split_manager.original_dataset.labels[dataset.indices])

    def train(self, num_epochs):
        if self.args.check_untrained_accuracy:
            for epoch, name, load_model in [(-1, "-1", True), (0, "0", False)]:
                self.eval_model(epoch, name,  splits_to_eval=self.args.splits_to_eval, load_model=load_model, skip_eval_if_already_done=self.args.skip_eval_if_already_done)
                self.record_keeper.save_records()
        self.split_manager.set_curr_split("train", is_training=True)
        self.training_assertions(self.trainer)        
        self.trainer.train(self.epoch, num_epochs)
        self.epoch = self.trainer.epoch + 1

    def eval(self):
        best_epoch, _ = self.hooks.get_best_epoch_and_accuracy(self.tester_obj, "val", ignore_epoch=(-1,0))
        eval_dict = {"best": best_epoch}
        if self.args.check_untrained_accuracy: eval_dict["-1"] = -1
        for name, epoch in eval_dict.items():
            self.eval_model(epoch, name, splits_to_eval=self.args.splits_to_eval, load_model=True, skip_eval_if_already_done=self.args.skip_eval_if_already_done)
            self.record_keeper.save_records()

    def meta_ConcatenateEmbeddings(self, model_suffix): 
        list_of_trunks, list_of_embedders = [], []
        for split_scheme_name in self.split_manager.split_scheme_names:
            self.split_manager.set_curr_split_scheme(split_scheme_name)
            self.set_curr_folders()
            trunk_model, embedder_model = self.load_model_for_eval(suffix=model_suffix)
            list_of_trunks.append(trunk_model.module)
            list_of_embedders.append(embedder_model.module)
        embedder_input_sizes = [self.base_model_output_size] * len(list_of_trunks)
        if isinstance(embedder_input_sizes[0], list):
            embedder_input_sizes = [np.sum(x) for x in embedder_input_sizes]
        operation_before_concat = (lambda x: torch.nn.functional.normalize(x, p=2, dim=1)) if self.args.eval_normalize_embeddings else None
        trunk = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_trunks))
        embedder = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_embedders, embedder_input_sizes, operation_before_concat))
        return trunk, embedder

    def meta_eval(self):
        assert self.args.splits_to_eval == ["test"]
        meta_model_getter = getattr(self, "meta_"+self.args.meta_testing_method)
        self.models = {}
        self.record_keeper = self.meta_record_keeper
        self.hooks = logging_presets.HookContainer(self.record_keeper, record_group_name_prefix=meta_model_getter.__name__, primary_metric=self.args.eval_primary_metric)
        self.tester_obj = self.pytorch_getter.get("tester", 
                                                self.args.testing_method, 
                                                self.get_tester_kwargs())

        eval_dict = {"best": 1}
        if self.args.check_untrained_accuracy: eval_dict["-1"] = -1

        group_name = self.get_eval_record_name_dict("meta_ConcatenateEmbeddings")["test"]
        len_of_existing_records = c_f.try_getting_db_count(self.meta_record_keeper, group_name)

        for name, i in eval_dict.items():
            self.models["trunk"], self.models["embedder"] = meta_model_getter(name)
            self.set_transforms()
            did_not_skip = self.eval_model(i, name, splits_to_eval=self.args.splits_to_eval, load_model=False, skip_eval_if_already_done=self.args.skip_meta_eval_if_already_done)
            if did_not_skip:
                is_trained = int(i==1)
                global_iteration = len_of_existing_records + is_trained + 1
                self.record_keeper.update_records({"is_trained": int(i==1)}, global_iteration=global_iteration, input_group_name_for_non_objects=group_name)
                self.record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=global_iteration, input_group_name_for_non_objects=group_name)

                for irrelevant_key in ["epoch", "best_epoch", "best_accuracy"]:
                    self.record_keeper.record_writer.records.pop(irrelevant_key, None)
                self.record_keeper.save_records()


    def get_eval_record_name_dict(self, eval_type="non_meta", return_all=False):
        prefix = self.hooks.record_group_name_prefix 
        self.hooks.record_group_name_prefix = "" #temporary
        non_meta = {k:self.hooks.record_group_name(self.tester_obj, k) for k in ["train", "val", "test"]}
        meta = {k:"meta_"+v for k,v in non_meta.items()}
        meta_concatenated = {k:"meta_ConcatenateEmbeddings_"+v for k,v in non_meta.items()}
        self.hooks.record_group_name_prefix = prefix

        name_dict = {"non_meta": non_meta,
                    "meta": meta,
                    "meta_ConcatenateEmbeddings": meta_concatenated}

        if return_all:
            return name_dict
        return name_dict[eval_type]




