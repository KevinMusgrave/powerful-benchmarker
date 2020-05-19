from ..utils import dataset_utils as d_u
from .index_split_manager import IndexSplitManager
from collections import OrderedDict

class ClassDisjointSplitManager(IndexSplitManager):

    def get_attribute_that_should_be_disjoint(self, dataset):
        return d_u.get_labels_by_hierarchy(d_u.get_subset_dataset_labels(dataset), self.hierarchy_level)

    def get_list_for_split_scheme_rule(self, dataset):
        labels = d_u.get_labels_by_hierarchy(dataset.labels, self.hierarchy_level)
        return sorted(list(set(labels)))

    def get_rule_input(self, dataset):
        return dataset.labels