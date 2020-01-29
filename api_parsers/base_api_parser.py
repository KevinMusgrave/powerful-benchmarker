import sys
from easy_module_attribute_getter import PytorchGetter
import datasets
import copy
from utils import common_functions as c_f, split_manager
from pytorch_metric_learning import trainers, losses, miners, regularizers, samplers, testers
import pytorch_metric_learning.utils.logging_presets as logging_presets
import pytorch_metric_learning.utils.common_functions as pml_cf
from torch.utils.tensorboard import SummaryWriter
import torch.nn
import torch
import architectures
import os
import shutil
import logging
import numpy as np
from scipy import stats as scipy_stats
from collections import defaultdict
import inspect

class BaseAPIParser:
    def __init__(self, args):
        os.environ["TORCH_HOME"] = args.pytorch_home
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.experiment_folder = args.experiment_folder
        self.pytorch_getter = PytorchGetter(use_pretrainedmodels_package=True)
        self.pytorch_getter.register('model', architectures.misc_models)
        self.pytorch_getter.register('loss', losses)
        self.pytorch_getter.register('miner', miners)
        self.pytorch_getter.register('regularizer', regularizers)
        self.pytorch_getter.register('sampler', samplers)
        self.pytorch_getter.register('trainer', trainers)
        self.pytorch_getter.register('tester', testers)
        self.pytorch_getter.register('dataset', datasets)

        _model_folder = os.path.join("%s", "%s", "saved_models")
        _pkl_folder = os.path.join("%s", "%s", "saved_pkls")
        _tensorboard_folder = os.path.join("%s", "%s", "tensorboard_logs")
        self.sub_experiment_dirs = [
            _model_folder,
            _pkl_folder,
            _tensorboard_folder
        ]

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
                    return mean, sem if hasattr(self, "meta_record_keeper") else mean
                else:
                    raise ValueError
            self.record_meta_logs()
        self.flush_tensorboard()
        if self.is_training():
            return self.return_val_accuracy_and_standard_error()

    def run_for_each_split_scheme(self):
        for split_scheme_name in self.split_manager.split_scheme_names:
            num_epochs = self.num_epochs[split_scheme_name]
            self.split_manager.set_curr_split_scheme(split_scheme_name)
            self.set_curr_folders()
            self.set_models_optimizers_losses()
            if self.args.evaluate:
                self.eval()
            elif self.should_train(num_epochs, split_scheme_name):
                self.train(num_epochs)
            self.update_meta_record_keeper(split_scheme_name)

    def is_training(self):
        return not self.args.evaluate

    def beginning_of_training(self):
        return (not self.args.resume_training) and (not self.args.evaluate)

    def make_dir(self):
        if os.path.isdir(self.experiment_folder):
            logging.info("Experiment name already taken!")
            sys.exit()
        c_f.makedir_if_not_there(self.experiment_folder)

    def make_sub_experiment_dirs(self):
        for s in self.sub_experiment_dirs:
            for r in self.split_manager.split_scheme_names:
                c_f.makedir_if_not_there(s % (self.experiment_folder, r))

    def set_curr_folders(self):
        self.model_folder, self.pkl_folder, self.tensorboard_folder = self.get_sub_experiment_dir_paths()[self.split_manager.curr_split_scheme_name]

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
            if s is not None: self.lr_schedulers[k + "_scheduler"] = s
            if g is not None: self.gradient_clippers[k + "_grad_clipper"] = g

    def set_split_manager(self):
        chosen_dataset = self.pytorch_getter.get("dataset", yaml_dict=self.args.dataset, additional_params={"dataset_root":self.args.dataset_root})
        self.split_manager = split_manager.SplitManager(dataset=chosen_dataset, 
                                                        test_size=self.args.test_size,
                                                        test_start_idx=self.args.test_start_idx, 
                                                        num_training_partitions=self.args.num_training_partitions,
                                                        num_training_sets=self.args.num_training_sets,
                                                        special_split_scheme_name=self.args.special_split_scheme_name,
                                                        hierarchy_level=self.args.label_hierarchy_level)

    def get_transforms(self):
        try:
            trunk = self.models["trunk"]
            if isinstance(trunk, torch.nn.DataParallel):
                trunk = trunk.module
            model_transform_properties = {k:getattr(trunk, k) for k in ["mean", "std", "input_space", "input_range"]}
        except:
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
            if "loss_function" in miner_params:
                loss_args = miner_params["loss_function"]
                miner_params["loss_function"] = self.get_loss_function(loss_args)
            if "mining_function" in miner_params:
                miner_args = miner_params["mining_function"]
                miner_params["mining_function"] = self.get_mining_function(miner_args)
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
        if "num_classes" in str(inspect.signature(loss.__init__)):
            loss_params["num_classes"] = self.split_manager.get_num_labels()
        if "regularizer" in loss_params:
            loss_params["regularizer"] = self.pytorch_getter.get("regularizer", yaml_dict=loss_params["regularizer"])
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
        if splits_to_eval is None or any(x in ["train", "val"] for x in splits_to_eval):
            return ["test"]
        return []

    def eval_assertions(self, dataset_dict):
        for v in dataset_dict.values():
            assert v.dataset.transform is self.split_manager.eval_transform

    def eval_model(self, epoch, suffix, splits_to_eval=None, load_model=False, **kwargs):
        logging.info("Launching evaluation for epoch %d"%epoch)
        if load_model:
            trunk_model, embedder_model = self.load_model_for_eval(suffix=suffix)
        else:
            trunk_model, embedder_model = self.models["trunk"], self.models["embedder"]
        trunk_model, embedder_model = trunk_model.to(self.device), embedder_model.to(self.device)
        splits_to_exclude = self.get_splits_exclusion_list(splits_to_eval)
        dataset_dict = self.split_manager.get_dataset_dict(exclusion_list=splits_to_exclude, is_training=False)
        self.eval_assertions(dataset_dict)
        self.hooks.run_tester_separately(self.tester_obj, dataset_dict, epoch, trunk_model, embedder_model, splits_to_eval, **kwargs)

    def flush_tensorboard(self):
        for writer in ["tensorboard_writer", "meta_tensorboard_writer"]:
            w = getattr(self, writer, None)
            if w:
                w.flush()
                w.close()

    def set_record_keeper(self):
        self.record_keeper, self.pickler_and_csver, self.tensorboard_writer = logging_presets.get_record_keeper(self.pkl_folder, self.tensorboard_folder)

    def set_meta_record_keeper(self):
        if len(self.split_manager.split_scheme_names) > 1:
            _, pkl_folder, tensorboard_folder = [s % (self.experiment_folder, "meta_logs") for s in self.sub_experiment_dirs]
            self.meta_record_keeper, self.meta_pickler_and_csver, self.meta_tensorboard_writer = logging_presets.get_record_keeper(pkl_folder, tensorboard_folder)
            self.meta_accuracies = defaultdict(lambda: defaultdict(dict))
            c_f.makedir_if_not_there(pkl_folder)
            c_f.makedir_if_not_there(tensorboard_folder)

    def update_meta_record_keeper(self, split_scheme_name):
        if hasattr(self, "meta_accuracies"):
            for split in self.args.splits_to_eval:
                best_split_accuracy = self.hooks.get_best_epoch_and_accuracy(self.tester_obj, split)[1]
                if best_split_accuracy is not None:
                    self.meta_accuracies[split]["average_best"][split_scheme_name] = best_split_accuracy
                untrained_accuracy = self.hooks.get_accuracy_of_epoch(self.tester_obj, split, -1)
                if untrained_accuracy is not None:
                    self.meta_accuracies[split]["untrained"][split_scheme_name] = untrained_accuracy

    def record_meta_logs(self):
        if hasattr(self, "meta_accuracies") and len(self.meta_accuracies) > 0:
            for split in self.args.splits_to_eval:
                group_name = "meta_" + self.hooks.record_group_name(self.tester_obj, split)
                averages = {"%s_%s"%(k, self.args.eval_metric_for_best_epoch): np.mean(list(v.values())) for k, v in self.meta_accuracies[split].items()}
                standard_errors = {"SEM_%s_%s"%(k, self.args.eval_metric_for_best_epoch): scipy_stats.sem(list(v.values())) for k, v in self.meta_accuracies[split].items()}
                len_of_existing_record = len(self.meta_record_keeper.get_record(group_name)[list(averages.keys())[0]])
                self.meta_record_keeper.update_records(averages, global_iteration=len_of_existing_record, input_group_name_for_non_objects=group_name)
                self.meta_record_keeper.update_records(standard_errors, global_iteration=len_of_existing_record, input_group_name_for_non_objects=group_name)
            self.meta_pickler_and_csver.save_records()

    def return_val_accuracy_and_standard_error(self):
        if hasattr(self, "meta_record_keeper"):
            group_name = "meta_" + self.hooks.record_group_name(self.tester_obj, "val")
            mean = self.meta_record_keeper.get_record(group_name)["average_best_%s"%self.args.eval_metric_for_best_epoch][-1]
            standard_error = self.meta_record_keeper.get_record(group_name)["SEM_average_best_%s"%self.args.eval_metric_for_best_epoch][-1]
            return mean, standard_error
        return self.hooks.get_best_epoch_and_accuracy(self.tester_obj, "val")[1]

    def maybe_load_models_and_records(self):
        if hasattr(self, "meta_pickler_and_csver"):
            self.meta_pickler_and_csver.load_records()
        return self.hooks.load_latest_saved_models_and_records(self.trainer, self.model_folder, self.device)

    def set_models_optimizers_losses(self):
        self.set_model()
        self.set_transforms()
        self.set_sampler()
        self.set_loss_function()
        self.set_mining_function()
        self.set_optimizers()
        self.set_record_keeper()
        self.hooks = logging_presets.HookContainer(self.record_keeper)
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
            "metric_for_best_epoch": self.args.eval_metric_for_best_epoch,
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
        splits_to_exclude = self.get_splits_exclusion_list(None)
        dataset_dict = self.split_manager.get_dataset_dict(exclusion_list=splits_to_exclude, is_training=False)
        helper_hook = self.hooks.end_of_epoch_hook(tester=self.tester_obj,
                                                    dataset_dict=dataset_dict,
                                                    model_folder=self.model_folder,
                                                    test_interval=self.args.save_interval,
                                                    validation_split_name="val",
                                                    patience=self.args.patience)
        def end_of_epoch_hook(trainer):
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
        best_epoch, curr_accuracy = self.hooks.get_best_epoch_and_curr_accuracy(self.tester_obj, "val", self.epoch)
        if best_epoch is None:
            return True
        else:
            return self.hooks.patience_remaining(self.epoch, best_epoch, self.args.patience) and self.latest_sub_experiment_epochs[split_scheme_name] < num_epochs

    def training_assertions(self, trainer):
        assert trainer.dataset is self.split_manager.curr_split_scheme["train"][0]
        assert trainer.dataset.dataset.transform is self.split_manager.train_transform

    def train(self, num_epochs):
        if self.epoch == 1 and self.args.check_untrained_accuracy:
            self.eval_model(-1, "-1", load_model=True)
        self.split_manager.set_curr_split("train", is_training=True)
        self.training_assertions(self.trainer)        
        self.trainer.train(self.epoch, num_epochs)
        self.epoch = self.trainer.epoch + 1

    def eval(self, **kwargs):
        best_epoch = self.hooks.get_best_epoch_and_accuracy(self.tester_obj, "val")[0]
        for name, epoch in {"-1": -1, "best": best_epoch}.items():
            self.eval_model(epoch, name, splits_to_eval=self.args.splits_to_eval, load_model=True, **kwargs)
        self.pickler_and_csver.save_records()

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

    def meta_eval(self, **kwargs):
        assert self.args.splits_to_eval == ["test"]
        meta_model_getter = getattr(self, "meta_"+self.args.meta_testing_method)
        self.models = {}
        self.record_keeper = self.meta_record_keeper
        self.pickler_and_csver = self.meta_pickler_and_csver
        self.pickler_and_csver.load_records()
        self.hooks = logging_presets.HookContainer(self.record_keeper, record_group_name_prefix=meta_model_getter.__name__)
        self.tester_obj = self.pytorch_getter.get("tester", 
                                                self.args.testing_method, 
                                                self.get_tester_kwargs())
        group_name = self.hooks.record_group_name(self.tester_obj, "test")
        curr_records = self.meta_record_keeper.get_record(group_name)
        iteration = len(list(curr_records.values())[0]) - 1 if len(curr_records) > 0 else 0 #this abomination is necessary
        for name, i in {"-1": -1, "best": iteration}.items():
            self.models["trunk"], self.models["embedder"] = meta_model_getter(name)
            self.set_transforms()
            self.eval_model(i, name, splits_to_eval=self.args.splits_to_eval, load_model=False, **kwargs)
        self.pickler_and_csver.save_records()
