from ..utils import common_functions as c_f, dataset_utils as d_u, constants as const
import pytorch_metric_learning.utils.common_functions as pml_cf
import os
import logging
from .getter_and_setter import GetterAndSetter
from .folder_creator import FolderCreator

class BaseAPIParser(GetterAndSetter, FolderCreator):
    def run(self):
        if self.beginning_of_training():
            self.make_dir()
        self.transforms = self.get_transforms()
        self.split_manager = self.get_split_manager()
        self.save_config_files()
        self.set_num_epochs_dict()
        self.make_sub_experiment_dirs()
        self.run_train_or_eval()


    def run_train_or_eval(self):
        if self.args.evaluate_ensemble:
            self.eval_ensemble()
        else:
            self.meta_record_keeper = self.get_meta_record_keeper()
            self.aggregator = self.get_aggregator()
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
        hooks, tester = self.get_dummy_hook_and_tester()
        self.aggregator.record_accuracies(self.args.splits_to_eval, self.meta_record_keeper, hooks, tester)


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


    def eval_model(self, epoch, model_name, hooks, tester, models=None, load_model=False, skip_eval_if_already_done=True):
        logging.info("Launching evaluation for model %s"%model_name)
        if load_model:
            logging.info("Initializing/loading models for evaluation")
            trunk_model, embedder_model = c_f.load_model_for_eval(self.factories["model"], self.args.models, model_name, factory_kwargs=self.all_kwargs("model"), model_folder=self.model_folder, device=self.device)
        else:
            logging.info("Using input models for evaluation")
            trunk_model, embedder_model = models["trunk"], models["embedder"]
        trunk_model, embedder_model = trunk_model.to(self.device), embedder_model.to(self.device)
        dataset_dict = self.split_manager.get_dataset_dict("eval", inclusion_list=self.args.splits_to_eval)
        self.eval_assertions(dataset_dict)
        return hooks.run_tester_separately(tester, dataset_dict, epoch, trunk_model, embedder_model, 
                                        splits_to_eval=self.args.splits_to_eval, collate_fn=self.split_manager.collate_fn, skip_eval_if_already_done=skip_eval_if_already_done)


    def eval_ensemble(self):
        ensemble = self.get_ensemble()
        record_keeper = self.get_meta_record_keeper()
        hooks = self.factories["hook"].create(named_specs={"hook_container": self.args.hook_container},
                                                subset="hook_container", 
                                                kwargs={"record_keeper": record_keeper, "record_group_name_prefix": ensemble.__class__.__name__})
        tester = self.get_tester()

        models_to_eval = []
        if self.args.check_untrained_accuracy: 
            models_to_eval.append(const.UNTRAINED_TRUNK)
            models_to_eval.append(const.UNTRAINED_TRUNK_AND_EMBEDDER)
        models_to_eval.append(const.TRAINED)

        group_names = ensemble.get_eval_record_name_dict(hooks, tester, self.args.splits_to_eval)
        models = {}
        for name in models_to_eval:
            split_folders = [x["models"] for x in [self.get_sub_experiment_dir_paths()[y] for y in self.split_manager.split_scheme_names]]
            list_of_trunks, list_of_embedders = ensemble.get_list_of_models(self.factories["model"], 
                                                                            self.args.models, 
                                                                            name,
                                                                            factory_kwargs=self.all_kwargs("model"),
                                                                            model_folder=split_folders, 
                                                                            device=self.device)
            models["trunk"], models["embedder"] = ensemble.create_ensemble_model(list_of_trunks, list_of_embedders)
            did_not_skip = self.eval_model(name, name, hooks, tester, models=models, skip_eval_if_already_done=self.args.skip_ensemble_eval_if_already_done)
            if did_not_skip:
                for group_name in group_names:
                    len_of_existing_records = c_f.try_getting_db_count(record_keeper, group_name) + 1
                    record_keeper.update_records({const.TRAINED_STATUS_COL_NAME: name}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                    record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)

                for irrelevant_key in ["best_epoch", "best_accuracy"]:
                    record_keeper.record_writer.records[group_name].pop(irrelevant_key, None)
                record_keeper.save_records()


    def should_train(self, num_epochs, split_scheme_name):
        best_epoch, _ = pml_cf.latest_version(self.model_folder, best=True)
        return self.hooks.patience_remaining(self.epoch, best_epoch, self.args.patience) and self.latest_sub_experiment_epochs[split_scheme_name] < num_epochs

    def training_assertions(self, trainer):
        assert trainer.dataset is self.split_manager.get_dataset("train", "train")
        assert d_u.get_underlying_dataset(trainer.dataset).transform == self.transforms["train"]

    def eval_assertions(self, dataset_dict):
        for k, v in dataset_dict.items():
            assert v is self.split_manager.get_dataset("eval", k)
            assert d_u.get_underlying_dataset(v).transform is self.transforms["eval"]

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

    def get_eval_record_name_dict(self, meta_names=(), return_with_split_names=True):
        hooks, tester = self.get_dummy_hook_and_tester()
        split_names = self.split_manager.split_names if return_with_split_names else None
        output = c_f.get_eval_record_name_dict(hooks, tester, split_names=split_names)
        for mn in meta_names:
            output[mn] = {k:"{}_{}".format(mn, v) for k,v in output.items()}
        return output

    def is_training(self):
        return (not self.args.evaluate) and (not self.args.evaluate_ensemble)

    def beginning_of_training(self):
        return (not self.args.resume_training) and self.is_training()

    def set_num_epochs_dict(self):
        if isinstance(self.args.num_epochs_train, int):
            self.num_epochs = {k: self.args.num_epochs_train for k in self.split_manager.split_scheme_names}
        else:
            self.num_epochs = self.args.num_epochs_train
