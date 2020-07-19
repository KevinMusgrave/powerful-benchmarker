import sys
import copy
from ..utils import common_functions as c_f, dataset_utils as d_u, constants as const
import pytorch_metric_learning.utils.logging_presets as logging_presets
import pytorch_metric_learning.utils.common_functions as pml_cf
from easy_module_attribute_getter import utils as emag_utils
import torch.nn
import torch
import os
import pathlib
import shutil
import logging
import numpy as np
from collections import defaultdict
from .. import architectures
from ..factories import FactoryFactory

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

        self.trainer, self.tester = None, None
        self.factories = FactoryFactory(api_parser=self, getter=self.pytorch_getter).create(named_specs=self.args.factories)


    def run(self):
        if self.beginning_of_training():
            self.make_dir()
        self.set_transforms()
        self.set_split_manager()
        self.save_config_files()
        self.set_num_epochs_dict()
        self.make_sub_experiment_dirs()
        self.run_train_or_eval()


    def run_train_or_eval(self):
        if self.args.evaluate_ensemble:
            self.eval_ensemble()
        else:
            self.set_meta_record_keeper()
            self.set_aggregator()
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
        self.delete_old_objects()
        if self.is_training():
            return self.aggregator.get_accuracy_and_standard_error(self.hooks, self.tester, self.meta_record_keeper, self.split_manager.num_split_schemes, "val")


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
            self.aggregator.update_accuracies(split_scheme_name, self.args.splits_to_eval, self.hooks, self.tester)
            self.delete_old_objects()
        hooks, _, tester = self.dummy_objs()
        self.aggregator.record_accuracies(self.args.splits_to_eval, self.meta_record_keeper, hooks, tester)


    def delete_old_objects(self):
        self.flush_tensorboard()
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


    def flush_tensorboard(self):
        for keeper in ["record_keeper", "meta_record_keeper"]:
            k = getattr(self, keeper, None)
            if k:
                k.tensorboard_writer.flush()
                k.tensorboard_writer.close()


    def is_training(self):
        return (not self.args.evaluate) and (not self.args.evaluate_ensemble)


    def beginning_of_training(self):
        return (not self.args.resume_training) and self.is_training()


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


    def save_config_files(self):
        self.latest_sub_experiment_epochs = c_f.latest_sub_experiment_epochs(self.get_sub_experiment_dir_paths())
        latest_epochs = list(self.latest_sub_experiment_epochs.values())
        if self.is_training():
            c_f.save_config_files(self.args.place_to_save_configs, self.args.dict_of_yamls, self.args.resume_training, latest_epochs)
        delattr(self.args, "dict_of_yamls")
        delattr(self.args, "place_to_save_configs")


    def set_num_epochs_dict(self):
        if isinstance(self.args.num_epochs_train, int):
            self.num_epochs = {k: self.args.num_epochs_train for k in self.split_manager.split_scheme_names}
        else:
            self.num_epochs = self.args.num_epochs_train


    def set_optimizers(self):
        self.optimizers, self.lr_schedulers, self.gradient_clippers = self.factories["optimizer"].create(named_specs=self.args.optimizers)
        
    def set_transforms(self):
        self.transforms = self.factories["transform"].create(named_specs=self.args.transforms)

    def set_split_manager(self):
        self.split_manager = self.factories["split_manager"].create(self.args.split_manager)

    def set_sampler(self):
        self.sampler = self.factories["sampler"].create(self.args.sampler)
               
    def set_loss_function(self):
        self.loss_funcs = self.factories["loss"].create(named_specs=self.args.loss_funcs)

    def set_mining_function(self):
        self.mining_funcs = self.factories["miner"].create(named_specs=self.args.mining_funcs)

    def set_model(self):
        self.models = self.factories["model"].create(named_specs=self.args.models)

    def set_tester(self):
        self.tester = self.factories["tester"].create(self.args.tester)

    def set_trainer(self):
        self.trainer = self.factories["trainer"].create(self.args.trainer)

    def set_record_keeper(self):
        self.record_keeper = self.factories["record_keeper"].create_record_keeper()

    def set_meta_record_keeper(self):
        self.meta_record_keeper = self.factories["record_keeper"].create_meta_record_keeper()

    def set_aggregator(self):
        self.aggregator = self.factories["aggregator"].create(self.args.aggregator)

    def set_hooks(self):
        self.hooks = self.factories["hook"].create_hook_container(self.args.hook_container)

    def set_dataparallel(self):
        for k, v in self.models.items():
            self.models[k] = torch.nn.DataParallel(v)

    def set_devices(self):
        for obj_dict in [self.models, self.loss_funcs, self.mining_funcs]:
            for v in obj_dict.values():
                v.to(self.device)
        for v in self.optimizers.values():
            c_f.move_optimizer_to_gpu(v, self.device)


    def eval_assertions(self, dataset_dict):
        for k, v in dataset_dict.items():
            assert v is self.split_manager.get_dataset("eval", k)
            assert d_u.get_underlying_dataset(v).transform is self.transforms["eval"]


    def eval_model(self, epoch, model_name, hooks, tester, models=None, load_model=False, skip_eval_if_already_done=True):
        logging.info("Launching evaluation for model %s"%model_name)
        if load_model:
            logging.info("Initializing/loading models for evaluation")
            trunk_model, embedder_model = c_f.load_model_for_eval(self.factories["model"], self.args.models, model_name, model_folder=self.model_folder, device=self.device)
        else:
            logging.info("Using input models for evaluation")
            trunk_model, embedder_model = models["trunk"], models["embedder"]
        trunk_model, embedder_model = trunk_model.to(self.device), embedder_model.to(self.device)
        dataset_dict = self.split_manager.get_dataset_dict("eval", inclusion_list=self.args.splits_to_eval)
        self.eval_assertions(dataset_dict)
        return hooks.run_tester_separately(tester, dataset_dict, epoch, trunk_model, embedder_model, 
                                        splits_to_eval=self.args.splits_to_eval, collate_fn=self.split_manager.collate_fn, skip_eval_if_already_done=skip_eval_if_already_done)


    def set_models_optimizers_losses(self):
        self.set_model()
        self.set_sampler()
        self.set_loss_function()
        self.set_mining_function()
        self.set_optimizers()
        self.set_record_keeper()
        self.set_hooks()
        self.set_tester()
        self.set_trainer()
        if self.is_training():
            self.epoch = self.hooks.load_latest_saved_models(self.trainer, self.model_folder, self.device, best=self.args.resume_training=="best")
        self.set_dataparallel()
        self.set_devices()


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
            eval_dict = self.get_eval_dict(False, True, True, randomize_embedder = self.epoch!=1)
            for name, (epoch, load_model) in eval_dict.items():
                self.eval_model(epoch, name, self.hooks, self.tester, models=self.models, skip_eval_if_already_done=self.args.skip_eval_if_already_done)
                self.record_keeper.save_records()
        self.training_assertions(self.trainer)        
        self.trainer.train(self.epoch, num_epochs)

    def eval(self):
        untrained = self.args.check_untrained_accuracy
        eval_dict = self.get_eval_dict(True, untrained, untrained, randomize_embedder=True)
        for name, (epoch, load_model) in eval_dict.items():
            self.eval_model(epoch, name, self.hooks, self.tester, models=self.models, load_model=load_model, skip_eval_if_already_done=self.args.skip_eval_if_already_done)
            self.record_keeper.save_records()

    def eval_ensemble(self):
        ensemble = self.factories["ensemble"].create(self.args.ensemble)
        models = {}
        self.record_keeper = self.factories["record_keeper"].create_meta_record_keeper()
        self.hooks = self.factories["hook"].create_hook_container(self.args.hook_container, record_group_name_prefix=ensemble.__class__.__name__)
        tester = self.factories["tester"].create(self.args.tester)

        models_to_eval = []
        if self.args.check_untrained_accuracy: 
            models_to_eval.append(const.UNTRAINED_TRUNK)
            models_to_eval.append(const.UNTRAINED_TRUNK_AND_EMBEDDER)
        models_to_eval.append(const.TRAINED)

        group_names = ensemble.get_eval_record_name_dict(self.hooks, tester, self.args.splits_to_eval)

        for name in models_to_eval:
            split_folders = [x["models"] for x in [self.get_sub_experiment_dir_paths()[y] for y in self.split_manager.split_scheme_names]]
            models["trunk"], models["embedder"] = ensemble.get_trunk_and_embedder(self.factories["model"], 
                                                                                    self.args.models, 
                                                                                    name, 
                                                                                    split_folders, 
                                                                                    self.device)
            did_not_skip = self.eval_model(name, name, self.hooks, tester, models=models, skip_eval_if_already_done=self.args.skip_ensemble_eval_if_already_done)
            if did_not_skip:
                for group_name in group_names:
                    len_of_existing_records = c_f.try_getting_db_count(self.record_keeper, group_name) + 1
                    self.record_keeper.update_records({const.TRAINED_STATUS_COL_NAME: name}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                    self.record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)

                for irrelevant_key in ["best_epoch", "best_accuracy"]:
                    self.record_keeper.record_writer.records[group_name].pop(irrelevant_key, None)
                self.record_keeper.save_records()


    def get_eval_record_name_dict(self, meta_names=(), return_with_split_names=True):
        hooks, split_manager, tester = self.dummy_objs()
        split_names = split_manager.split_names if return_with_split_names else None
        output = c_f.get_eval_record_name_dict(hooks, tester, split_names=split_names)
        for mn in meta_names:
            output[mn] = {k:"{}_{}".format(mn, v) for k,v in output.items()}
        return output


    def dummy_objs(self):
        hooks = self.factories["hook"].create_hook_container(self.args.hook_container, record_keeper=None)
        split_manager = self.factories["split_manager"].create(self.args.split_manager)
        tester = self.factories["tester"].create(self.args.tester)
        return hooks, split_manager, tester

