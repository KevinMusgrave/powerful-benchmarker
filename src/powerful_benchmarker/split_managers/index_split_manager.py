from ..utils import dataset_utils as d_u
from .base_split_manager import BaseSplitManager
import itertools
from collections import OrderedDict

class IndexSplitManager(BaseSplitManager):
    def __init__(self,
        test_size,
        test_start_idx,
        num_training_partitions,
        num_training_sets,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.test_size = test_size
        self.test_start_idx = test_start_idx
        self.num_training_partitions = num_training_partitions
        self.num_training_sets = num_training_sets

    def get_attribute_that_should_be_disjoint(self, dataset):
        return dataset.indices

    def assert_splits_are_disjoint(self):
        for (split_scheme_name, split_scheme) in self.split_schemes.items():
            for dataset_dict in split_scheme.values():
                should_be_disjoint = []
                for split, dataset in dataset_dict.items():
                    should_be_disjoint.append(set(self.get_attribute_that_should_be_disjoint(dataset)))
                for (x,y) in itertools.combinations(should_be_disjoint, 2):
                    assert x.isdisjoint(y)


    def _create_split_schemes(self, datasets):
        for partition in range(self.num_training_sets):
            name = self.get_split_name(partition)
            self.split_schemes[name] = OrderedDict()
            for train_or_eval, dataset_dict in datasets.items():
                self.split_schemes[name][train_or_eval] = self.create_one_split_scheme(dataset_dict, partition)


    def split_assertions(self):
        super().split_assertions()
        self.assert_splits_are_disjoint()

    def get_list_for_split_scheme_rule_creation(self, dataset):
        return list(range(len(dataset)))

    def get_rule_input(self, dataset):
        return self.get_list_for_split_scheme_rule_creation(dataset)

    def create_one_split_scheme(self, datasets, partition):
        sample_dataset = datasets[list(datasets.keys())[0]]
        traintest_dict = OrderedDict()
        list_for_rule_creation = self.get_list_for_split_scheme_rule(sample_dataset)
        rule_input = self.get_rule_input(sample_dataset)
        len_of_list_for_rule_creation = len(list_for_rule_creation)

        val_ratio = (1./self.num_training_partitions)*(1-self.test_size)
        train_ratio = (1. - val_ratio)*(1-self.test_size)
        class_ratios = {"train": train_ratio, "val": val_ratio, "test": self.test_size}
        split_lengths = d_u.split_lengths_from_ratios(class_ratios, len_of_list_for_rule_creation)

        test_set_rule = d_u.get_single_wrapped_range_rule(int(self.test_start_idx*len_of_list_for_rule_creation), split_lengths["test"], list_for_rule_creation)
        traintest_dict["test"] = d_u.create_rule_based_subset(datasets["test"], rule_input, test_set_rule)
        split_lengths.pop("test", None)
        exclusion_rule = lambda input_scalar: not test_set_rule(input_scalar)
        list_for_rule_creation = [x for x in list_for_rule_creation if exclusion_rule(x)]
        len_of_list_for_rule_creation = len(list_for_rule_creation)

        start_idx = int((float(partition)/self.num_training_partitions)*len_of_list_for_rule_creation)
        rules = d_u.get_wrapped_range_rules(start_idx, split_lengths, list_for_rule_creation, exclusion_rule)
        for k, rule in rules.items():
            traintest_dict[k] = d_u.create_rule_based_subset(datasets[k], rule_input, rule)

        return traintest_dict


    def get_base_split_name(self):
        test_size = int(self.test_size*100)
        test_start_idx = int(self.test_start_idx*100)
        return 'Test%02d_%02d_Partitions%d_'%(test_size, test_start_idx, self.num_training_partitions)


    def get_split_name(self, partition):
        return "{}{}".format(self.get_base_split_name(), partition)
