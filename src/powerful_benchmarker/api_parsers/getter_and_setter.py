from ..factories import FactoryFactory
import numpy as np
import torch
import os
import pytorch_metric_learning.utils.common_functions as pml_cf
import logging
from ..utils import common_functions as c_f
from easy_module_attribute_getter import utils as emag_utils

class GetterAndSetter:
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

        self.factories = FactoryFactory(getter=self.pytorch_getter).create(named_specs=self.args.factories)

    def set_models_optimizers_losses(self):
        self.models = self.get_model()
        self.sampler = self.get_sampler()
        self.loss_funcs = self.get_loss_funcs()
        self.mining_funcs = self.get_mining_funcs()
        self.optimizers, self.lr_schedulers, self.gradient_clippers = self.get_optimizers()
        self.record_keeper = self.get_record_keeper()
        self.hooks = self.get_hooks()
        self.tester = self.get_tester()
        self.trainer = self.get_trainer()
        if self.is_training():
            self.epoch = self.hooks.load_latest_saved_models(self.trainer, self.model_folder, self.device, best=self.args.resume_training=="best")
        self.set_dataparallel(self.models)
        self.set_devices(self.models, self.loss_funcs, self.mining_funcs, self.optimizers)

    def default_kwargs_optimizers(self):
        return {"param_sources": lambda: [self.models, self.loss_funcs]}

    def default_kwargs_transforms(self):
        return {"trunk_model": lambda: self.get_model()["trunk"]}

    def default_kwargs_split_manager(self):
        dataset, dataset_params = self.pytorch_getter.get("dataset", yaml_dict=self.args.dataset, return_uninitialized=True)
        return {"dataset": lambda: dataset,
                "original_dataset_params": lambda: dataset_params,
                "transforms": lambda: self.transforms,
                "dataset_root": lambda: self.args.dataset_root}

    def default_kwargs_sampler(self):
        return {"labels": lambda: self.split_manager.get_labels("train", "train"),
                "length_before_new_iter": lambda: len(self.split_manager.get_dataset("train", "train"))}

    def default_kwargs_loss_funcs(self):
        return {"num_classes": lambda: self.split_manager.get_num_labels("train", "train")}

    def default_kwargs_tester(self):
        return {"data_device": lambda: self.device,
                "data_and_label_getter": lambda: self.split_manager.data_and_label_getter,
                "end_of_testing_hook": lambda: self.hooks.end_of_testing_hook} 

    def default_kwargs_trainer(self):
        return {
            "models": lambda: self.models,
            "optimizers": lambda: self.optimizers,
            "sampler": lambda: self.sampler,
            "collate_fn": lambda: self.split_manager.collate_fn,
            "loss_funcs": lambda: self.loss_funcs,
            "mining_funcs": lambda: self.mining_funcs,
            "dataset": lambda: self.split_manager.get_dataset("train", "train", log_split_details=True),
            "data_device": lambda: self.device,
            "lr_schedulers": lambda: self.lr_schedulers,
            "gradient_clippers": lambda: self.gradient_clippers,
            "data_and_label_getter": lambda: self.split_manager.data_and_label_getter,
            "dataset_labels": lambda: list(self.split_manager.get_label_set("train", "train")),
            "end_of_iteration_hook": lambda: self.hooks.end_of_iteration_hook,
            "end_of_epoch_hook": lambda: self.factories["hook"].create(named_specs={"end_of_epoch_hook": None}, subset="end_of_epoch_hook", **self.all_kwargs("end_of_epoch_hook", {}))
        }   

    def default_kwargs_record_keeper(self):
        return {"csv_folder": lambda: self.csv_folder, 
                "tensorboard_folder": lambda: self.tensorboard_folder, 
                "global_db_path": lambda: self.global_db_path, 
                "experiment_name": lambda: self.args.experiment_name, 
                "is_new_experiment": lambda: self.beginning_of_training() and self.curr_split_count == 0, 
                "save_figures": lambda: self.args.save_figures_on_tensorboard,
                "save_lists": lambda: self.args.save_lists_in_db}

    def default_kwargs_meta_record_keeper(self):
        folders = {folder_type: s % (self.experiment_folder, "meta_logs") for folder_type, s in self.sub_experiment_dirs.items()}
        csv_folder, tensorboard_folder = folders["csvs"], folders["tensorboard"]
        return {"csv_folder": lambda: csv_folder, 
                "tensorboard_folder": lambda: tensorboard_folder,
                "global_db_path": lambda: self.global_db_path, 
                "experiment_name": lambda: self.args.experiment_name, 
                "is_new_experiment": lambda: self.beginning_of_training(),
                "save_figures": lambda: self.args.save_figures_on_tensorboard,
                "save_lists": lambda: self.args.save_lists_in_db}
    
    def default_kwargs_hooks(self):
        return {"record_keeper": lambda: self.record_keeper}

    def default_kwargs_end_of_epoch_hook(self):
        return {"hooks": lambda: self.hooks, 
                "split_manager": lambda: self.split_manager, 
                "splits_to_eval": lambda: self.args.splits_to_eval, 
                "tester": lambda: self.tester,
                "model_folder": lambda: self.model_folder, 
                "save_interval": lambda: self.args.save_interval, 
                "patience": lambda: self.args.patience, 
                "collate_fn": lambda: self.split_manager.collate_fn, 
                "eval_assertions": lambda: self.eval_assertions}

    def get_optimizers(self, **kwargs):
        return self.factories["optimizer"].create(named_specs=self.args.optimizers, **self.all_kwargs("optimizers", kwargs))
        
    def get_transforms(self, **kwargs):
        return self.factories["transform"].create(named_specs=self.args.transforms, **self.all_kwargs("transforms", kwargs))

    def get_split_manager(self, **kwargs):
        return self.factories["split_manager"].create(self.args.split_manager, **self.all_kwargs("split_manager", kwargs))

    def get_sampler(self, **kwargs):
        return self.factories["sampler"].create(self.args.sampler, **self.all_kwargs("sampler", kwargs))
               
    def get_loss_funcs(self, **kwargs):
        return self.factories["loss"].create(named_specs=self.args.loss_funcs, **self.all_kwargs("loss_funcs", kwargs))

    def get_mining_funcs(self, **kwargs):
        return self.factories["miner"].create(named_specs=self.args.mining_funcs, **self.all_kwargs("mining_funcs", kwargs))

    def get_model(self, **kwargs):
        return self.factories["model"].create(named_specs=self.args.models, **self.all_kwargs("model", kwargs))

    def get_tester(self, **kwargs):     
        return self.factories["tester"].create(self.args.tester, **self.all_kwargs("tester", kwargs))

    def get_trainer(self, **kwargs):     
        return self.factories["trainer"].create(self.args.trainer, **self.all_kwargs("trainer", kwargs))

    def get_record_keeper(self, **kwargs):
        return self.factories["record_keeper"].create(self._kwargs("record_keeper", kwargs))

    def get_meta_record_keeper(self, **kwargs):
        return self.factories["record_keeper"].create(self._kwargs("meta_record_keeper", kwargs))

    def get_aggregator(self, **kwargs):
        return self.factories["aggregator"].create(self.args.aggregator, **self.all_kwargs("aggregator", kwargs))

    def get_ensemble(self, **kwargs):
        return self.factories["ensemble"].create(self.args.ensemble, **self.all_kwargs("ensemble", kwargs))

    def get_hooks(self, **kwargs):
        return self.factories["hook"].create(named_specs={"hook_container": self.args.hook_container}, subset="hook_container", **self.all_kwargs("hooks", kwargs))

    def get_dummy_hook_and_tester(self):
        hooks = self.get_hooks(record_keeper=lambda: None)
        tester = self.get_tester(**self.nullify_kwargs(self.default_kwargs_tester()))
        return hooks, tester

    def set_dataparallel(self, models):
        for k, v in models.items():
            models[k] = torch.nn.DataParallel(v)

    def set_devices(self, models, losses, miners, optimizers):
        for obj_dict in [models, losses, miners]:
            for v in obj_dict.values():
                v.to(self.device)
        for v in optimizers.values():
            c_f.move_optimizer_to_gpu(v, self.device)

    def delete_old_objects(self):
        self.flush_tensorboard()
        for attr_name in self.list_of_old_objects_to_delete():
            try:
                delattr(self, attr_name)
                logging.info("Deleted self.%s"%attr_name)
            except AttributeError:
                pass

    def list_of_old_objects_to_delete(self):
        return ["models", "sampler", "loss_funcs", "mining_funcs", "optimizers", "lr_schedulers", "gradient_clippers", "record_keeper", "hooks", "trainer", "tester"]


    def flush_tensorboard(self):
        for keeper in ["record_keeper", "meta_record_keeper"]:
            k = getattr(self, keeper, None)
            if k:
                k.tensorboard_writer.flush()
                k.tensorboard_writer.close()


    def _kwargs(self, object_type, kwargs, prefix="default_kwargs"):
        default_kwargs_getter = getattr(self, "{}_{}".format(prefix, object_type), None)
        if default_kwargs_getter is None:
            if len(kwargs) == 0:
                return None
            final_kwargs = emag_utils.merge_two_dicts({}, kwargs)
        else:
            final_kwargs = emag_utils.merge_two_dicts(default_kwargs_getter(), kwargs)
        return {k: v() for k,v in final_kwargs.items()}

    def _per_name_kwargs(self, object_type, kwargs):
        return self._kwargs(object_type, kwargs, prefix="default_per_name_kwargs")

    def all_kwargs(self, object_type, kwargs=None, per_name_kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        per_name_kwargs = {} if per_name_kwargs is None else per_name_kwargs
        return {"kwargs": self._kwargs(object_type, kwargs), 
                "per_name_kwargs": self._per_name_kwargs(object_type, per_name_kwargs)}

    def nullify_kwargs(self, kwargs):
        return {k: lambda: None for k in kwargs.keys()}