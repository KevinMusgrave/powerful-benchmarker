import sys
import record_keeper as record_keeper_package
from easy_module_attribute_getter import PytorchGetter
import datasets
import copy
from utils import common_functions as c_f, split_manager
from pytorch_metric_learning import trainers, losses, miners, samplers, testers
from torch.utils.tensorboard import SummaryWriter
import torch.nn
import torch
import architectures
import os
import shutil
import logging
import numpy as np
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
        self.pytorch_getter.register('sampler', samplers)
        self.pytorch_getter.register('trainer', trainers)
        self.pytorch_getter.register('tester', testers)
        self.pytorch_getter.register('dataset', datasets)

        _model_folder = "%s/%s/saved_models"
        _pkl_folder = "%s/%s/saved_pkls"
        _tensorboard_folder = "%s/%s/tensorboard_logs"
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
            self.run_for_each_split_scheme()
            self.record_meta_logs()
        self.flush_tensorboard()
        return self.return_val_accuracy()

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
            self.sampler = self.pytorch_getter.get("sampler", yaml_dict=self.args.sampler, additional_params={"labels_to_indices":self.split_manager.labels_to_indices})

    def get_loss_function(self, loss_type):
        loss, loss_params = self.pytorch_getter.get("loss", yaml_dict=loss_type, return_uninitialized=True)
        loss_params = copy.deepcopy(loss_params)
        if "num_classes" in str(inspect.signature(loss.__init__)):
            loss_params["num_classes"] = self.split_manager.get_num_labels(self.args.label_hierarchy_level)
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
        trunk_model = torch.nn.DataParallel(self.get_trunk_model(self.args.models["trunk"]))
        if not untrained:
            embedder_model = torch.nn.DataParallel(self.get_embedder_model(self.args.models["embedder"], self.base_model_output_size))
            c_f.load_dict_of_models(
                {"trunk": trunk_model, "embedder": embedder_model},
                suffix,
                self.model_folder,
                self.device
            )
        else:
            embedder_model = torch.nn.DataParallel(architectures.misc_models.Identity())
        return trunk_model, embedder_model

    def get_splits_exclusion_list(self, splits_to_eval):
        if splits_to_eval is None or any(x in ["train", "val"] for x in splits_to_eval):
            return ["test"]
        return []

    def eval_assertions(self, dataset_dict):
        for v in dataset_dict.values():
            assert v.dataset.transform is self.split_manager.eval_transform

    def eval_model(self, epoch, suffix, splits_to_eval=None, load_model=False, **kwargs):
        if load_model:
            trunk_model, embedder_model = self.load_model_for_eval(suffix=suffix)
        else:
            trunk_model, embedder_model = self.models["trunk"], self.models["embedder"]
        trunk_model, embedder_model = trunk_model.to(self.device), embedder_model.to(self.device)
        splits_to_exclude = self.get_splits_exclusion_list(splits_to_eval)
        dataset_dict = self.split_manager.get_dataset_dict(exclusion_list=splits_to_exclude, is_training=False)
        self.eval_assertions(dataset_dict)
        self.tester_obj.test(dataset_dict, epoch, trunk_model, embedder_model, splits_to_eval, **kwargs)

    def save_stuff(self, curr_suffix, prev_suffix=None):
        for obj_dict in [self.models, self.optimizers, self.lr_schedulers, self.loss_funcs]:
            c_f.save_dict_of_models(obj_dict, curr_suffix, self.model_folder)
            if prev_suffix is not None:
                c_f.delete_dict_of_models(obj_dict, prev_suffix, self.model_folder) 

    def save_stuff_and_maybe_eval(self):
        if self.epoch % self.args.save_interval == 0:
            self.save_stuff(self.epoch, self.epoch-self.args.save_interval)
            if self.epoch == self.args.save_interval and self.args.check_untrained_accuracy:
                self.eval_model(-1, "-1", load_model=True)
                self.save_stuff("best")
            self.eval_model(self.epoch, str(self.epoch))
            self.set_best_epoch_and_curr_accuracy()
            if self.epoch == self.best_epoch:
                logging.info("New best accuracy!")
                self.save_stuff("best")
            self.pickler_and_csver.save_records()

    def flush_tensorboard(self):
        for writer in ["tensorboard_writer", "meta_tensorboard_writer"]:
            w = getattr(self, writer, None)
            if w:
                w.flush()
                w.close()

    def get_record_keeper(self, pkl_folder, tensorboard_folder):
        pickler_and_csver = record_keeper_package.PicklerAndCSVer(pkl_folder)
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_folder)
        record_keeper = record_keeper_package.RecordKeeper(tensorboard_writer, pickler_and_csver, ["record_these", "learnable_param_names"])
        return record_keeper, pickler_and_csver, tensorboard_writer

    def set_record_keeper(self):
        self.record_keeper, self.pickler_and_csver, self.tensorboard_writer = self.get_record_keeper(self.pkl_folder, self.tensorboard_folder)

    def set_meta_record_keeper(self):
        if len(self.split_manager.split_scheme_names) > 1:
            _, pkl_folder, tensorboard_folder = [s % (self.experiment_folder, "meta_logs") for s in self.sub_experiment_dirs]
            self.meta_record_keeper, self.meta_pickler_and_csver, self.meta_tensorboard_writer = self.get_record_keeper(pkl_folder, tensorboard_folder)
            self.meta_accuracies = defaultdict(lambda: defaultdict(dict))
            c_f.makedir_if_not_there(pkl_folder)
            c_f.makedir_if_not_there(tensorboard_folder)

    def update_meta_record_keeper(self, split_scheme_name):
        if hasattr(self, "meta_accuracies"):
            for split in self.args.splits_to_eval:
                best_split_accuracy = self.tester_obj.get_best_epoch_and_accuracy(split)[1]
                if best_split_accuracy is not None:
                    self.meta_accuracies[split]["average_best"][split_scheme_name] = best_split_accuracy
                untrained_accuracy = self.tester_obj.get_accuracy_of_epoch(split, -1)
                if untrained_accuracy is not None:
                    self.meta_accuracies[split]["untrained"][split_scheme_name] = untrained_accuracy

    def record_meta_logs(self):
        if hasattr(self, "meta_accuracies") and len(self.meta_accuracies) > 0:
            for split in self.args.splits_to_eval:
                group_name = "meta_" + self.tester_obj.record_group_name(split)
                averages = {"%s_%s"%(k, self.args.eval_metric_for_best_epoch): np.mean(list(v.values())) for k, v in self.meta_accuracies[split].items()}
                len_of_existing_record = len(self.meta_record_keeper.get_record(group_name)[list(averages.keys())[0]])
                self.meta_record_keeper.update_records(averages, global_iteration=len_of_existing_record, input_group_name_for_non_objects=group_name)
            self.meta_pickler_and_csver.save_records()

    def return_val_accuracy(self):
        if hasattr(self, "meta_record_keeper"):
            group_name = "meta_" + self.tester_obj.record_group_name("val")
            return self.meta_record_keeper.get_record(group_name)["average_best_%s"%self.args.eval_metric_for_best_epoch][-1]
        return self.tester_obj.get_best_epoch_and_accuracy("val")[1]

    def maybe_load_models_and_records(self):
        resume_epoch = 0
        self.pickler_and_csver.load_records()
        if hasattr(self, "meta_pickler_and_csver"):
            self.meta_pickler_and_csver.load_records()
        if self.args.resume_training:
            resume_epoch = c_f.latest_version(self.model_folder, "/trunk_*.pth") or 0
            if resume_epoch > 0:
                for obj_dict in [self.models, self.optimizers, self.lr_schedulers, self.loss_funcs]:
                    c_f.load_dict_of_models(obj_dict, resume_epoch, self.model_folder, self.device)
        return resume_epoch

    def set_models_optimizers_losses(self):
        self.set_model()
        self.set_transforms()
        self.set_sampler()
        self.set_dataparallel()
        self.set_loss_function()
        self.set_mining_function()
        self.set_optimizers()
        self.set_record_keeper()
        self.tester_obj = self.pytorch_getter.get("tester", self.args.testing_method, self.get_tester_kwargs())
        self.epoch = self.maybe_load_models_and_records() + 1

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
            "record_keeper": self.record_keeper,
            "data_and_label_getter": lambda data: (data["data"], data["label"]),
            "label_hierarchy_level": self.args.label_hierarchy_level 
        }

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
            "record_keeper": self.record_keeper,
            "iterations_per_epoch": self.args.iterations_per_epoch,
            "label_mapper": self.split_manager.map_labels,
            "lr_schedulers": self.lr_schedulers,
            "gradient_clippers": self.gradient_clippers,
            "freeze_trunk_batchnorm": self.args.freeze_batchnorm,
            "label_hierarchy_level": self.args.label_hierarchy_level,
            "dataloader_num_workers": self.args.dataloader_num_workers,
            "loss_weights": getattr(self.args, "loss_weights", None),
            "data_and_label_getter": lambda data: (data["data"], data["label"]) 
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

    def set_best_epoch_and_curr_accuracy(self):
        self.curr_accuracy = self.tester_obj.get_accuracy_of_epoch('val', self.epoch)
        self.best_epoch = self.tester_obj.get_best_epoch_and_accuracy('val')[0]

    def patience_remaining(self):
        if self.args.patience is not None:
            if self.epoch - self.best_epoch > self.args.patience:
                logging.info("Validation accuracy has plateaued. Exiting.")
                return False
        return True

    def should_train(self, num_epochs, split_scheme_name):
        self.set_best_epoch_and_curr_accuracy()
        if self.best_epoch is None:
            return True
        else:
            return self.patience_remaining() and self.latest_sub_experiment_epochs[split_scheme_name] < num_epochs

    def training_assertions(self):
        assert self.trainer.dataset is self.split_manager.curr_split_scheme["train"][0]
        assert self.trainer.dataset.dataset.transform is self.split_manager.train_transform

    def train(self, num_epochs):
        self.trainer = self.pytorch_getter.get("trainer", self.args.training_method, self.get_trainer_kwargs())
        while self.epoch <= num_epochs:
            self.split_manager.set_curr_split("train", is_training=True)
            self.training_assertions()
            self.set_devices()
            self.trainer.train(self.epoch, self.args.save_interval)
            self.epoch = self.trainer.epoch
            self.save_stuff_and_maybe_eval()
            if not self.patience_remaining():
                return
            self.trainer.step_lr_plateau_schedulers(self.curr_accuracy)
            self.epoch += 1

    def eval(self, **kwargs):
        best_epoch = -1
        for group_name, value in self.record_keeper.pickler_and_csver.records.items():
            if group_name.startswith("accuracies") and group_name.endswith("VAL"):
                best_epoch = max(best_epoch, value["best_epoch"][-1])
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
        self.tester_obj = self.pytorch_getter.get("tester", 
                                                self.args.testing_method, 
                                                self.get_tester_kwargs(), 
                                                additional_params={"record_group_name_prefix": meta_model_getter.__name__})
        group_name = self.tester_obj.record_group_name("test")
        curr_records = self.meta_record_keeper.get_record(group_name)
        iteration = len(list(list(curr_records.values())[0].values())) - 1 if len(curr_records) > 0 else 0 #this abomination is necessary
        for name, i in {"-1": -1, "best": iteration}.items():
            self.models["trunk"], self.models["embedder"] = meta_model_getter(name)
            self.set_transforms()
            self.eval_model(i, name, splits_to_eval=self.args.splits_to_eval, load_model=False, **kwargs)
        self.pickler_and_csver.save_records()
