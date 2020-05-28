from ..utils import dataset_utils as d_u
from .index_split_manager import IndexSplitManager
import numpy as np

class ClassDisjointSplitManager(IndexSplitManager):

    def get_list_for_splitting(self, dataset):
        return sorted(list(self.get_label_set(dataset=dataset)))

    def convert_to_subset_idx(self, dataset, split_from_kfolder):
        return np.where(np.isin(self.get_labels(dataset=dataset), split_from_kfolder))[0]

    def get_list_for_class_disjoint_assertion(self, dataset):
        return self.get_list_for_splitting(dataset)

    def class_disjoint_assertion(self):
        for t_type in self.split_scheme_holder.get_transform_types():
            self.assert_across("split_scheme_names", 
                                "disjoint", 
                                within_group=True, 
                                attribute_descriptor="class labels",
                                attribute_getter=self.get_list_for_class_disjoint_assertion, 
                                transform_types=[t_type], 
                                split_names=self.split_scheme_holder.get_split_names())

    def split_assertions(self):
        super().split_assertions()
        self.class_disjoint_assertion()
