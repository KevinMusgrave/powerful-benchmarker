from .base_factory import BaseFactory
import pytorch_metric_learning.utils.logging_presets as logging_presets

class RecordKeeperFactory(BaseFactory):
    def create_record_keeper(self, *_):
        is_new_experiment = self.api_parser.beginning_of_training() and self.api_parser.curr_split_count == 0
        record_keeper, _, _ = logging_presets.get_record_keeper(csv_folder = self.api_parser.csv_folder, 
                                                            tensorboard_folder = self.api_parser.tensorboard_folder, 
                                                            global_db_path = self.api_parser.global_db_path, 
                                                            experiment_name = self.api_parser.args.experiment_name, 
                                                            is_new_experiment = is_new_experiment, 
                                                            save_figures = self.api_parser.args.save_figures_on_tensorboard,
                                                            save_lists = self.api_parser.args.save_lists_in_db)
        return record_keeper


    def create_meta_record_keeper(self, *_):
        is_new_experiment = self.api_parser.beginning_of_training()
        folders = {folder_type: s % (self.api_parser.experiment_folder, "meta_logs") for folder_type, s in self.api_parser.sub_experiment_dirs.items()}
        csv_folder, tensorboard_folder = folders["csvs"], folders["tensorboard"]
        meta_record_keeper, _, _ = logging_presets.get_record_keeper(csv_folder = csv_folder, 
                                                                        tensorboard_folder = tensorboard_folder,
                                                                        global_db_path = self.api_parser.global_db_path, 
                                                                        experiment_name = self.api_parser.args.experiment_name, 
                                                                        is_new_experiment = is_new_experiment,
                                                                        save_figures = self.api_parser.args.save_figures_on_tensorboard,
                                                                        save_lists = self.api_parser.args.save_lists_in_db)
        return meta_record_keeper