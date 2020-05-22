from ..utils import dataset_utils as d_u
from .index_split_manager import IndexSplitManager
from sklearn.model_selection import train_test_split, StratifiedKFold

class ClosedSetSplitManager(IndexSplitManager):
    def get_trainval_and_test(self, dataset, input_list):
        trainval_set, test_set, _, _ = train_test_split(input_list, 
                                                        input_list, 
                                                        test_size=self.test_size, 
                                                        shuffle=True, 
                                                        random_state=self.random_seed, 
                                                        stratify=dataset.labels)
        return trainval_set, test_set

    def get_kfold_generator(self, dataset, trainval_set):
        kfolder = StratifiedKFold(n_splits=self.num_training_partitions, shuffle=self.shuffle, random_state=self.random_seed)
        return kfolder.split(trainval_set, self.get_labels(dataset=dataset)[trainval_set])