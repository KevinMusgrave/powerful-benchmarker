from ..factories import FactoryFactory
import numpy as np
import torch
import os
import pytorch_metric_learning.utils.common_functions as pml_cf
import logging
from ..utils import common_functions as c_f

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

        self.trainer, self.tester = None, None
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

    def kwargs_optimizers(self):
        return {"param_sources": [self.models, self.loss_funcs]}

    def kwargs_transforms(self):
        return {"trunk_model": self.get_model()["trunk"]}

    def kwargs_split_manager(self):
        dataset, dataset_params = self.pytorch_getter.get("dataset", yaml_dict=self.args.dataset, return_uninitialized=True)
        return {"dataset": dataset,
                "original_dataset_params": dataset_params,
                "transforms": self.transforms,
                "dataset_root": self.args.dataset_root}

    def kwargs_sampler(self):
        return {"labels": self.split_manager.get_labels("train", "train"),
                "length_before_new_iter": len(self.split_manager.get_dataset("train", "train"))}

    def kwargs_loss_funcs(self):
        return {"num_classes": self.split_manager.get_num_labels("train", "train")}

    def kwargs_tester(self):
        return {"data_device": self.device,
                "data_and_label_getter": self.split_manager.data_and_label_getter,
                "end_of_testing_hook": self.hooks.end_of_testing_hook} 

    def kwargs_trainer(self):
        end_of_epoch_hook = self.factories["hook"].create(named_specs={"end_of_epoch_hook": None}, subset="end_of_epoch_hook", **self.all_kwargs("end_of_epoch_hook"))
        return {
            "models": self.models,
            "optimizers": self.optimizers,
            "sampler": self.sampler,
            "collate_fn": self.split_manager.collate_fn,
            "loss_funcs": self.loss_funcs,
            "mining_funcs": self.mining_funcs,
            "dataset": self.split_manager.get_dataset("train", "train", log_split_details=True),
            "data_device": self.device,
            "lr_schedulers": self.lr_schedulers,
            "gradient_clippers": self.gradient_clippers,
            "data_and_label_getter": self.split_manager.data_and_label_getter,
            "dataset_labels": list(self.split_manager.get_label_set("train", "train")),
            "end_of_iteration_hook": self.hooks.end_of_iteration_hook,
            "end_of_epoch_hook": end_of_epoch_hook
        }   

    def kwargs_record_keeper(self):
        return {"csv_folder": self.csv_folder, 
                "tensorboard_folder": self.tensorboard_folder, 
                "global_db_path": self.global_db_path, 
                "experiment_name": self.args.experiment_name, 
                "is_new_experiment": self.beginning_of_training() and self.curr_split_count == 0, 
                "save_figures": self.args.save_figures_on_tensorboard,
                "save_lists": self.args.save_lists_in_db}

    def kwargs_meta_record_keeper(self):
        folders = {folder_type: s % (self.experiment_folder, "meta_logs") for folder_type, s in self.sub_experiment_dirs.items()}
        csv_folder, tensorboard_folder = folders["csvs"], folders["tensorboard"]
        return {"csv_folder": csv_folder, 
                "tensorboard_folder": tensorboard_folder,
                "global_db_path": self.global_db_path, 
                "experiment_name": self.args.experiment_name, 
                "is_new_experiment": self.beginning_of_training(),
                "save_figures": self.args.save_figures_on_tensorboard,
                "save_lists": self.args.save_lists_in_db}
    
    def kwargs_hooks(self):
        return {"record_keeper": self.record_keeper}

    def kwargs_end_of_epoch_hook(self):
        return {"hooks": self.hooks, 
                "split_manager": self.split_manager, 
                "splits_to_eval": self.args.splits_to_eval, 
                "tester": self.tester,
                "model_folder": self.model_folder, 
                "save_interval": self.args.save_interval, 
                "patience": self.args.patience, 
                "collate_fn": self.split_manager.collate_fn, 
                "eval_assertions": self.eval_assertions}

    def get_optimizers(self):
        return self.factories["optimizer"].create(named_specs=self.args.optimizers, **self.all_kwargs("optimizers"))
        
    def get_transforms(self):
        return self.factories["transform"].create(named_specs=self.args.transforms, **self.all_kwargs("transforms"))

    def get_split_manager(self):
        return self.factories["split_manager"].create(self.args.split_manager, **self.all_kwargs("split_manager"))

    def get_sampler(self):
        return self.factories["sampler"].create(self.args.sampler, **self.all_kwargs("sampler"))
               
    def get_loss_funcs(self):
        return self.factories["loss"].create(named_specs=self.args.loss_funcs, **self.all_kwargs("loss_funcs"))

    def get_mining_funcs(self):
        return self.factories["miner"].create(named_specs=self.args.mining_funcs, **self.all_kwargs("mining_funcs"))

    def get_model(self):
        return self.factories["model"].create(named_specs=self.args.models, **self.all_kwargs("model"))

    def get_tester(self):     
        return self.factories["tester"].create(self.args.tester, **self.all_kwargs("tester"))

    def get_trainer(self):     
        return self.factories["trainer"].create(self.args.trainer, **self.all_kwargs("trainer"))

    def get_record_keeper(self):
        return self.factories["record_keeper"].create(self.kwargs_record_keeper())

    def get_meta_record_keeper(self):
        return self.factories["record_keeper"].create(self.kwargs_meta_record_keeper())

    def get_aggregator(self):
        return self.factories["aggregator"].create(self.args.aggregator, **self.all_kwargs("aggregator"))

    def get_ensemble(self):
        return self.factories["ensemble"].create(self.args.ensemble, **self.all_kwargs("ensemble"))

    def get_hooks(self):
        return self.factories["hook"].create(named_specs={"hook_container": self.args.hook_container}, subset="hook_container", **self.all_kwargs("hooks"))

    def get_dummy_hook_and_tester(self):
        hooks = self.factories["hook"].create(named_specs={"hook_container": self.args.hook_container},
                                                subset="hook_container", 
                                                kwargs={"record_keeper": None}) # no record_keeper
        tester = self.factories["tester"].create(self.args.tester) # no tester args
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


    def kwargs(self, object_type, prefix="kwargs"):
        kwargs_getter = getattr(self, "{}_{}".format(prefix, object_type), None)
        if kwargs_getter is None:
            return None
        return kwargs_getter()

    def kwargs_per_name(self, object_type):
        return self.kwargs(object_type, prefix="kwargs_per_name")

    def all_kwargs(self, object_type):
        return {"kwargs": self.kwargs(object_type), "kwargs_per_name": self.kwargs_per_name(object_type)}