import sys
import copy
from ..utils import common_functions as c_f, dataset_utils as d_u, constants as const
from pytorch_metric_learning import losses
import pytorch_metric_learning.utils.logging_presets as logging_presets
import pytorch_metric_learning.utils.common_functions as pml_cf
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
        self.sub_experiment_dirs = {
            "models": _model_folder,
            "csvs": _csv_folder,
            "tensorboard": _tensorboard_folder
        }

        self.trainer, self.tester_obj = None, None

    def run(self):
        if self.beginning_of_training():
            self.make_dir()
        self.set_split_manager()
        self.save_config_files()
        self.set_num_epochs_dict()
        self.make_sub_experiment_dirs()
        self.set_meta_record_keeper()
        if self.args.evaluate:
            return_dict = {}
            for self.curr_meta_testing_method in c_f.if_str_convert_to_singleton_list(self.args.meta_testing_method):
                return_dict[self.curr_meta_testing_method] = self.run_train_or_eval()
            return return_dict
        else:
            return self.run_train_or_eval()


    def run_train_or_eval(self):
        if self.args.evaluate and self.get_curr_meta_testing_method():
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
                    return mean, sem if self.split_manager.num_split_schemes > 1 else mean
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
        for s in self.sub_experiment_dirs.values():
            for r in self.split_manager.split_scheme_names:
                c_f.makedir_if_not_there(s % (self.experiment_folder, r))

    def set_curr_folders(self):
        folders = self.get_sub_experiment_dir_paths()[self.split_manager.curr_split_scheme_name]
        self.model_folder, self.csv_folder, self.tensorboard_folder = folders["models"], folders["csvs"], folders["tensorboard"]

    def get_sub_experiment_dir_paths(self):
        sub_experiment_dir_paths = {}
        for k in self.split_manager.split_scheme_names:
            sub_experiment_dir_paths[k] = {folder_type: s % (self.experiment_folder, k) for folder_type, s in self.sub_experiment_dirs.items()}
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
            c_f.save_config_files(self.args.place_to_save_configs, self.args.dict_of_yamls, self.args.resume_training, latest_epochs)
        delattr(self.args, "dict_of_yamls")
        delattr(self.args, "place_to_save_configs")

    def get_accuracy_calculator(self):
        return self.pytorch_getter.get("accuracy_calculator", yaml_dict=self.args.eval_accuracy_calculator)

    def get_collate_fn(self):
        return None

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
            if s is not None: self.lr_schedulers = {"%s_%s"%(basename, k):v for k,v in s.items()}
            if g is not None: self.gradient_clippers[basename + "_grad_clipper"] = g


    def get_split_manager(self, yaml_dict=None):
        yaml_dict = self.args.split_manager if yaml_dict is None else yaml_dict
        split_manager, split_manager_params = self.pytorch_getter.get("split_manager", yaml_dict=yaml_dict, return_uninitialized=True)
        split_manager_params = copy.deepcopy(split_manager_params)
        if c_f.check_init_arguments(split_manager, "model"):
            split_manager_params["model"] = torch.nn.DataParallel(self.get_trunk_model(self.args.models["trunk"])).to(self.device)
        if "helper_split_manager" in split_manager_params:
            split_manager_params["helper_split_manager"] = self.get_split_manager(yaml_dict=split_manager_params["helper_split_manager"])
        return split_manager(**split_manager_params)

    def set_split_manager(self):
        self.split_manager = self.get_split_manager()

        if self.args.multi_dataset is not None:
            chosen_dataset, original_dataset_params = {}, {}
            for split_name, yaml_dict in self.args.multi_dataset.items():
                chosen_dataset[split_name], original_dataset_params[split_name] = self.pytorch_getter.get("dataset", yaml_dict=yaml_dict, return_uninitialized=True)
            self.split_manager.split_names = list(self.args.multi_dataset.keys())
        else:
            chosen_dataset, original_dataset_params = self.pytorch_getter.get("dataset", yaml_dict=self.args.dataset, return_uninitialized=True)
            chosen_dataset = {k:chosen_dataset for k in self.split_manager.split_names}
            original_dataset_params = {k:original_dataset_params for k in self.split_manager.split_names}

        datasets = defaultdict(dict)
        for transform_type, T in self.get_transforms().items():
            logging.info("{} transform: {}".format(transform_type, T))
            for split_name in self.split_manager.split_names:
                dataset_params = copy.deepcopy(original_dataset_params[split_name])
                dataset_params["transform"] = T
                if "root" not in dataset_params:
                    dataset_params["root"] = self.args.dataset_root            
                datasets[transform_type][split_name] = chosen_dataset[split_name](**dataset_params)
        
        self.split_manager.create_split_schemes(datasets)


    def get_transforms(self):
        try:
            trunk = self.get_trunk_model(self.args.models["trunk"])
            if isinstance(trunk, torch.nn.DataParallel):
                trunk = trunk.module
            model_transform_properties = {k:getattr(trunk, k) for k in ["mean", "std", "input_space", "input_range"]}
        except (KeyError, AttributeError):
            model_transform_properties = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        self.transforms = {"train": None, "eval": None}
        for k, v in self.args.transforms.items():
            self.transforms[k] = self.pytorch_getter.get_composed_img_transform(v, **model_transform_properties)
        return self.transforms

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
            sampler, sampler_params = self.pytorch_getter.get("sampler", yaml_dict=self.args.sampler, return_uninitialized=True)
            if c_f.check_init_arguments(sampler, "labels"):
                sampler_params = copy.deepcopy(sampler_params)
                sampler_params["labels"] = self.split_manager.get_labels("train", "train")
            self.sampler = sampler(**sampler_params)
            

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
            loss_params["num_classes"] = self.split_manager.get_num_labels("train", "train")
            logging.info("Passing %d as num_classes to the loss function"%loss_params["num_classes"])
        if "regularizer" in loss_params:
            loss_params["regularizer"] = self.pytorch_getter.get("regularizer", yaml_dict=loss_params["regularizer"])
        if "num_class_per_param" in loss_params and loss_params["num_class_per_param"]:
            loss_params["num_class_per_param"] = self.split_manager.get_num_labels("train", "train")
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

    def load_model_for_eval(self, model_name):
        untrained_trunk = model_name in const.UNTRAINED_TRUNK_ALIASES
        untrained_trunk_and_embedder = model_name in const.UNTRAINED_TRUNK_AND_EMBEDDER_ALIASES
        trunk_model = self.get_trunk_model(self.args.models["trunk"])
        if untrained_trunk:
            embedder_model = architectures.misc_models.Identity()
        else:
            embedder_model = self.get_embedder_model(self.args.models["embedder"], self.base_model_output_size)
            if not untrained_trunk_and_embedder: 
                if model_name in const.TRAINED_ALIASES:
                    _, model_name = pml_cf.latest_version(self.model_folder, best=True)
                pml_cf.load_dict_of_models(
                    {"trunk": trunk_model, "embedder": embedder_model},
                    model_name,
                    self.model_folder,
                    self.device,
                    log_if_successful = True,
                    assert_success = True
                )
        return torch.nn.DataParallel(trunk_model), torch.nn.DataParallel(embedder_model)

    def eval_assertions(self, dataset_dict):
        for k, v in dataset_dict.items():
            assert v is self.split_manager.get_dataset("eval", k)
            assert d_u.get_underlying_dataset(v).transform is self.transforms["eval"]

    def eval_model(self, epoch, model_name, load_model=False, skip_eval_if_already_done=True):
        logging.info("Launching evaluation for model %s"%model_name)
        if load_model:
            trunk_model, embedder_model = self.load_model_for_eval(model_name=model_name)
        else:
            trunk_model, embedder_model = self.models["trunk"], self.models["embedder"]
        trunk_model, embedder_model = trunk_model.to(self.device), embedder_model.to(self.device)
        dataset_dict = self.split_manager.get_dataset_dict("eval", inclusion_list=self.args.splits_to_eval)
        self.eval_assertions(dataset_dict)
        return self.hooks.run_tester_separately(self.tester_obj, dataset_dict, epoch, trunk_model, embedder_model, 
                                        splits_to_eval=self.args.splits_to_eval, collate_fn=self.get_collate_fn(), skip_eval_if_already_done=skip_eval_if_already_done)

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
        folders = {folder_type: s % (self.experiment_folder, "meta_logs") for folder_type, s in self.sub_experiment_dirs.items()}
        csv_folder, tensorboard_folder = folders["csvs"], folders["tensorboard"]
        self.meta_record_keeper, _, _ = logging_presets.get_record_keeper(csv_folder, tensorboard_folder,  self.global_db_path, self.args.experiment_name, is_new_experiment)
        self.meta_accuracies = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def update_meta_record_keeper(self, split_scheme_name):
        for split in self.args.splits_to_eval:
            untrained_trunk_accuracies = self.hooks.get_accuracies_of_epoch(self.tester_obj, split, const.UNTRAINED_TRUNK_INT)
            untrained_trunk_embedder_accuracies = self.hooks.get_accuracies_of_epoch(self.tester_obj, split, const.UNTRAINED_TRUNK_AND_EMBEDDER_INT)
            best_split_accuracies, _ = self.hooks.get_accuracies_of_best_epoch(self.tester_obj, split, ignore_epoch=const.IGNORE_ALL_UNTRAINED)
            accuracies_dict = {const.UNTRAINED_TRUNK: untrained_trunk_accuracies, const.UNTRAINED_TRUNK_AND_EMBEDDER: untrained_trunk_embedder_accuracies, const.TRAINED: best_split_accuracies}
            for trained_status, accuracies in accuracies_dict.items():
                if len(accuracies) > 0:
                    accuracy_keys = [k for k in accuracies[0].keys() if any(acc in k for acc in self.tester_obj.accuracy_calculator.get_curr_metrics())]
                    for k in accuracy_keys:
                        self.meta_accuracies[split][trained_status][k][split_scheme_name] = accuracies[0][k]

    def record_meta_logs(self):
        if len(self.meta_accuracies) > 0:
            for split in self.args.splits_to_eval:
                group_name = self.get_eval_record_name_dict(const.META_SEPARATE_EMBEDDINGS)[split]
                len_of_existing_records = c_f.try_getting_db_count(self.meta_record_keeper, group_name)
                for trained_status, accuracies in self.meta_accuracies[split].items():
                    if len(accuracies) > 0:
                        len_of_existing_records += 1
                        averages = {k: np.mean(list(v.values())) for k, v in accuracies.items()}
                        self.meta_record_keeper.update_records(averages, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                        if self.split_manager.num_split_schemes > 1:
                            standard_errors = {"SEM_%s"%k: scipy_stats.sem(list(v.values())) for k, v in accuracies.items()}
                            self.meta_record_keeper.update_records(standard_errors, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                        self.meta_record_keeper.update_records({const.TRAINED_STATUS_COL_NAME: trained_status}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                        self.meta_record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
            self.meta_record_keeper.save_records()

    def return_val_accuracy_and_standard_error(self):
        group_name = self.get_eval_record_name_dict(const.META_SEPARATE_EMBEDDINGS)["val"]
        def get_average_best_and_sem(key):
            if self.split_manager.num_split_schemes > 1:
                sem_key = "SEM_%s"%key
                columns = "%s, %s"%(key, sem_key)
                return_keys = (key, sem_key)
            else:
                columns = key
                return_keys = (key, )
            query = "SELECT {0} FROM {1} WHERE {2}=? AND id=(SELECT MAX(id) FROM {1})".format(columns, group_name, const.TRAINED_STATUS_COL_NAME)
            return self.meta_record_keeper.query(query, values=(const.TRAINED,), use_global_db=False), return_keys
        q, keys = self.hooks.try_primary_metric(self.tester_obj, get_average_best_and_sem)
        if len(keys) > 1:
            return tuple(q[0][k] for k in keys)
        return q[0][keys[0]]

    def maybe_load_latest_saved_models(self):
        return self.hooks.load_latest_saved_models(self.trainer, self.model_folder, self.device, best=self.args.resume_training=="best")

    def set_models_optimizers_losses(self):
        self.set_model()
        self.set_sampler()
        self.set_loss_function()
        self.set_mining_function()
        self.set_optimizers()
        self.set_record_keeper()
        self.hooks = logging_presets.HookContainer(self.record_keeper, primary_metric=self.args.eval_primary_metric, validation_split_name="val")
        self.tester_obj = self.pytorch_getter.get("tester", self.args.testing_method, self.get_tester_kwargs())
        self.trainer = self.pytorch_getter.get("trainer", self.args.training_method, self.get_trainer_kwargs())
        if self.is_training():
            self.epoch = self.maybe_load_latest_saved_models()
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
            "data_and_label_getter": self.split_manager.data_and_label_getter,
            "label_hierarchy_level": self.args.label_hierarchy_level,
            "end_of_testing_hook": self.hooks.end_of_testing_hook,
            "accuracy_calculator": self.get_accuracy_calculator() 
        }

    def get_end_of_epoch_hook(self):
        logging.info("Creating end_of_epoch_hook kwargs")
        dataset_dict = self.split_manager.get_dataset_dict("eval", inclusion_list=self.args.splits_to_eval)
        helper_hook = self.hooks.end_of_epoch_hook(tester=self.tester_obj,
                                                    dataset_dict=dataset_dict,
                                                    model_folder=self.model_folder,
                                                    test_interval=self.args.save_interval,
                                                    patience=self.args.patience,
                                                    test_collate_fn=self.get_collate_fn())
        def end_of_epoch_hook(trainer):
            torch.cuda.empty_cache()
            self.eval_assertions(dataset_dict)
            return helper_hook(trainer)

        return end_of_epoch_hook

    def get_trainer_kwargs(self):
        logging.info("Getting trainer kwargs")
        return {
            "models": self.models,
            "optimizers": self.optimizers,
            "batch_size": self.args.batch_size,
            "sampler": self.sampler,
            "collate_fn": self.get_collate_fn(),
            "loss_funcs": self.loss_funcs,
            "mining_funcs": self.mining_funcs,
            "dataset": self.split_manager.get_dataset("train", "train", log_split_details=True),
            "data_device": self.device,
            "iterations_per_epoch": self.args.iterations_per_epoch,
            "lr_schedulers": self.lr_schedulers,
            "gradient_clippers": self.gradient_clippers,
            "freeze_trunk_batchnorm": self.args.freeze_batchnorm,
            "label_hierarchy_level": self.args.label_hierarchy_level,
            "dataloader_num_workers": self.args.dataloader_num_workers,
            "loss_weights": getattr(self.args, "loss_weights", None),
            "data_and_label_getter": self.split_manager.data_and_label_getter,
            "dataset_labels": list(self.split_manager.get_label_set("train", "train")),
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
        best_epoch, _ = pml_cf.latest_version(self.model_folder, best=True)
        return self.hooks.patience_remaining(self.epoch, best_epoch, self.args.patience) and self.latest_sub_experiment_epochs[split_scheme_name] < num_epochs

    def training_assertions(self, trainer):
        assert trainer.dataset is self.split_manager.get_dataset("train", "train")
        assert d_u.get_underlying_dataset(trainer.dataset).transform == self.transforms["train"]

    def get_eval_dict(self, best, untrained_trunk, untrained_trunk_and_embedder, randomize_embedder):
        eval_dict = {}
        if untrained_trunk:
            eval_dict[const.UNTRAINED_TRUNK] = (const.UNTRAINED_TRUNK_INT, True)
        if untrained_trunk_and_embedder:
            eval_dict[const.UNTRAINED_TRUNK_AND_EMBEDDER] = (const.UNTRAINED_TRUNK_AND_EMBEDDER_INT, randomize_embedder)
        if best:
            best_epoch, _ = pml_cf.latest_version(self.model_folder, best=True)
            eval_dict["best"] = (best_epoch, True)
        return eval_dict

    def train(self, num_epochs):
        if self.args.check_untrained_accuracy:
            eval_dict = self.get_eval_dict(False, True, True, randomize_embedder = self.epoch==1)
            for name, (epoch, load_model) in eval_dict.items():
                self.eval_model(epoch, name, load_model=load_model, skip_eval_if_already_done=self.args.skip_eval_if_already_done)
                self.record_keeper.save_records()
        self.training_assertions(self.trainer)        
        self.trainer.train(self.epoch, num_epochs)

    def eval(self):
        untrained = self.args.check_untrained_accuracy
        eval_dict = self.get_eval_dict(True, untrained, untrained, randomize_embedder=True)
        for name, (epoch, load_model) in eval_dict.items():
            self.eval_model(epoch, name, load_model=load_model, skip_eval_if_already_done=self.args.skip_eval_if_already_done)
            self.record_keeper.save_records()

    def meta_ConcatenateEmbeddings(self, model_name): 
        list_of_trunks, list_of_embedders = [], []
        for split_scheme_name in self.split_manager.split_scheme_names:
            self.split_manager.set_curr_split_scheme(split_scheme_name)
            self.set_curr_folders()
            trunk_model, embedder_model = self.load_model_for_eval(model_name=model_name)
            list_of_trunks.append(trunk_model.module)
            list_of_embedders.append(embedder_model.module)
        embedder_input_sizes = [self.base_model_output_size] * len(list_of_trunks)
        if isinstance(embedder_input_sizes[0], list):
            embedder_input_sizes = [np.sum(x) for x in embedder_input_sizes]
        normalize_embeddings_func = lambda x: torch.nn.functional.normalize(x, p=2, dim=1)
        embedder_operation_before_concat = normalize_embeddings_func if self.args.eval_normalize_embeddings else None
        trunk_operation_before_concat = normalize_embeddings_func if self.args.eval_use_trunk_output else None

        trunk = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_trunks, operation_before_concat=trunk_operation_before_concat))
        embedder = torch.nn.DataParallel(architectures.misc_models.ListOfModels(list_of_embedders, embedder_input_sizes, embedder_operation_before_concat))
        return trunk, embedder

    def meta_eval(self):
        meta_model_getter = getattr(self, self.get_curr_meta_testing_method())
        self.models = {}
        self.record_keeper = self.meta_record_keeper
        self.hooks = logging_presets.HookContainer(self.record_keeper, record_group_name_prefix=meta_model_getter.__name__, primary_metric=self.args.eval_primary_metric)
        self.tester_obj = self.pytorch_getter.get("tester", 
                                                self.args.testing_method, 
                                                self.get_tester_kwargs())

        models_to_eval = []
        if self.args.check_untrained_accuracy: 
            models_to_eval.append(const.UNTRAINED_TRUNK)
            models_to_eval.append(const.UNTRAINED_TRUNK_AND_EMBEDDER)
        models_to_eval.append(const.TRAINED)

        group_names = [self.get_eval_record_name_dict(self.curr_meta_testing_method)[split_name] for split_name in self.args.splits_to_eval]

        for name in models_to_eval:
            self.models["trunk"], self.models["embedder"] = meta_model_getter(name)
            did_not_skip = self.eval_model(name, name, load_model=False, skip_eval_if_already_done=self.args.skip_meta_eval_if_already_done)
            if did_not_skip:
                for group_name in group_names:
                    len_of_existing_records = c_f.try_getting_db_count(self.meta_record_keeper, group_name) + 1
                    self.record_keeper.update_records({const.TRAINED_STATUS_COL_NAME: name}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                    self.record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)

                for irrelevant_key in ["best_epoch", "best_accuracy"]:
                    self.record_keeper.record_writer.records[group_name].pop(irrelevant_key, None)
                self.record_keeper.save_records()


    def get_eval_record_name_dict(self, eval_type=const.NON_META, return_all=False, return_base_record_group_name=False):
        if not getattr(self, "hooks", None):
            self.hooks = logging_presets.HookContainer(None, primary_metric=self.args.eval_primary_metric)
        if not getattr(self, "tester_obj", None):
            if not getattr(self, "split_manager", None):
                self.split_manager = self.get_split_manager()
            self.tester_obj = self.pytorch_getter.get("tester", self.args.testing_method, self.get_tester_kwargs())
        prefix = self.hooks.record_group_name_prefix 
        self.hooks.record_group_name_prefix = "" #temporary
        if return_base_record_group_name:
            non_meta = {"base_record_group_name": self.hooks.base_record_group_name(self.tester_obj)}
        else:
            non_meta = {k:self.hooks.record_group_name(self.tester_obj, k) for k in self.split_manager.split_names}
        meta_separate = {k:"{}_{}".format(const.META_SEPARATE_EMBEDDINGS, v) for k,v in non_meta.items()}
        meta_concatenate = {k:"{}_{}".format(const.META_CONCATENATE_EMBEDDINGS, v) for k,v in non_meta.items()}
        self.hooks.record_group_name_prefix = prefix

        name_dict = {const.NON_META: non_meta,
                    const.META_SEPARATE_EMBEDDINGS: meta_separate,
	                const.META_CONCATENATE_EMBEDDINGS: meta_concatenate}

        if return_all:
            return name_dict
        return name_dict[eval_type]


    def get_curr_meta_testing_method(self):
        # META_SEPARATE_EMBEDDINGS is equivalent to the regular per-split eval
        if self.curr_meta_testing_method == const.META_SEPARATE_EMBEDDINGS:
            return None
        return self.curr_meta_testing_method



