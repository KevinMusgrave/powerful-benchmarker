import pathlib
import sys
from ..utils import common_functions as c_f
import logging 

class FolderCreator:
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