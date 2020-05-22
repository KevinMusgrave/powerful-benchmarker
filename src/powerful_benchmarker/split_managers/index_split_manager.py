from ..utils import dataset_utils as d_u, common_functions as c_f
from .base_split_manager import BaseSplitManager
import itertools
from collections import OrderedDict
from sklearn.model_selection import train_test_split, KFold
import numpy as np

class IndexSplitManager(BaseSplitManager):
    def __init__(self,
        test_size,
        test_start_idx,
        num_training_partitions,
        num_training_sets,
        shuffle = False,
        random_seed = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.test_size = test_size
        self.test_start_idx = test_start_idx
        self.num_training_partitions = num_training_partitions
        self.num_training_sets = num_training_sets
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split_assertions(self):
        super().split_assertions()
        for t_type in self.split_scheme_holder.get_transform_types():
            self.assert_across("split_scheme_names", "disjoint", transform_types=[t_type], split_names=["val"])

    def get_trainval_and_test(self, dataset, input_list):
        trainval_set, test_set, _, _ = train_test_split(input_list, 
                                                        input_list, 
                                                        test_size=self.test_size, 
                                                        shuffle=self.shuffle, 
                                                        random_state=self.random_seed)
        return trainval_set, test_set

    def get_kfold_generator(self, dataset, trainval_set):
        kfolder = KFold(n_splits=self.num_training_partitions, shuffle=self.shuffle, random_state=self.random_seed)
        return kfolder.split(trainval_set)

    def get_list_for_splitting(self, dataset):
        return np.arange(len(dataset))

    def convert_to_subset_idx(self, dataset, split_from_kfolder):
        return split_from_kfolder

    def _create_split_schemes(self, datasets):
        # the assumption is all datasets are the same
        sample_dataset = c_f.first_val_of_dict(c_f.first_val_of_dict(datasets))
        list_for_splitting = self.get_list_for_splitting(sample_dataset)
        if not self.shuffle:
            num_idx = len(list_for_splitting)
            test_start = int(self.test_start_idx*num_idx)
            test_size = int(self.test_size*num_idx)
            list_for_splitting = np.roll(list_for_splitting, test_start - test_size)

        trainval_set, test_set = self.get_trainval_and_test(sample_dataset, list_for_splitting)
        split_schemes = OrderedDict()

        for i, (train_idx, val_idx) in enumerate(self.get_kfold_generator(sample_dataset, trainval_set)):
            split_dict = {"train": trainval_set[train_idx], "val": trainval_set[val_idx], "test": test_set}
            name = self.get_split_scheme_name(i)
            split_schemes[name] = OrderedDict()
            for transform_type in datasets.keys():
                split_schemes[name][transform_type] = OrderedDict()
                if i < self.num_training_sets:
                    for k, v in split_dict.items():
                        curr_dataset = datasets[transform_type][k]
                        subset_idx = self.convert_to_subset_idx(curr_dataset, v)
                        split_schemes[name][transform_type][k] = d_u.create_subset(curr_dataset, subset_idx)

        return split_schemes


    def get_base_split_scheme_name(self):
        test_size = int(self.test_size*100)
        test_start_idx = int(self.test_start_idx*100)
        return 'Test%02d_%02d_Partitions%d_'%(test_size, test_start_idx, self.num_training_partitions)


    def get_split_scheme_name(self, partition):
        return "{}{}".format(self.get_base_split_scheme_name(), partition)
