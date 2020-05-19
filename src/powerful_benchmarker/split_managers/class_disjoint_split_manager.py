from ..utils import dataset_utils as d_u
from .index_split_manager import IndexSplitManager
import numpy as np

class ClassDisjointSplitManager(IndexSplitManager):

    def get_list_for_splitting(self, dataset):
        label_set = d_u.get_label_set(dataset.labels, self.hierarchy_level)
        return sorted(list(label_set))

    def convert_to_subset_idx(self, dataset, split_from_kfolder):
        return np.where(np.isin(dataset.labels, split_from_kfolder))[0]

    def get_list_for_class_disjoint_assertion(self, dataset):
        labels = d_u.get_subset_dataset_labels(dataset)
        label_set = d_u.get_label_set(labels, self.hierarchy_level)
        return list(label_set)

    def split_assertions(self):
        super().split_assertions()
        for t_type in self.split_scheme_holder.get_transform_types():
            self.assert_across("split_scheme_names", 
                                "disjoint", 
                                within_group=True, 
                                attribute_descriptor="class labels",
                                attribute_getter=self.get_list_for_class_disjoint_assertion, 
                                transform_types=[t_type], 
                                split_names=self.split_scheme_holder.get_split_names())