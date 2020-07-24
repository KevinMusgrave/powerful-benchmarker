from .class_disjoint_split_manager import ClassDisjointSplitManager

class MLRCSplitManager(ClassDisjointSplitManager):

    def numeric_class_rule(self, a, b):
        if a < b:
            range_rule = lambda label: (a <= label <= b)
        elif a > b:
            range_rule = lambda label: (label >= a or label <= b)
        return range_rule

    def get_wrapped_range(self, start_idx, length_of_range, length_of_list):
        s = start_idx % length_of_list
        e = (start_idx + length_of_range - 1) % length_of_list
        return s, e

    def get_single_class_rule(self, start_idx, split_length, sorted_label_set):
        s, e = self.get_wrapped_range(start_idx, split_length, len(sorted_label_set))
        s = sorted_label_set[s]
        e = sorted_label_set[e]
        return self.numeric_class_rule(s, e)

    def get_class_rules(self, start_idx, split_lengths, sorted_label_set):
        class_rules = {}
        for k, v in split_lengths.items():
            class_rules[k] = self.get_single_class_rule(start_idx, v, sorted_label_set)
            start_idx += v
        return class_rules

    def get_kfold_generator(self, dataset, trainval_set):
        trainval_idx_tuples = []
        num_labels = len(trainval_set)
        self.split_lengths.pop("test", None)
        for partition in range(self.num_training_partitions):
            start_idx = int((float(partition)/self.num_training_partitions)*num_labels)
            class_rules = self.get_class_rules(start_idx, self.split_lengths, trainval_set)
            train_idx = [i for i,x in enumerate(trainval_set) if class_rules["train"](x)]
            val_idx = [i for i,x in enumerate(trainval_set) if class_rules["val"](x)]
            trainval_idx_tuples.append((train_idx, val_idx))
        return trainval_idx_tuples

    def get_trainval_and_test(self, dataset, input_list):
        ratios = {}
        val_ratio = (1./self.num_training_partitions)*(1-self.test_size)
        train_ratio = (1. - val_ratio)*(1-self.test_size)
        ratios = {"train": train_ratio, "val": val_ratio, "test": self.test_size}
        num_labels = len(input_list)
        self.split_lengths = {k: int(num_labels * v) for k,v in ratios.items()}
        self.split_lengths["train"] += num_labels - sum(v for k, v in self.split_lengths.items())
        assert sum(v for v in self.split_lengths.values()) == num_labels
        trainval_set = input_list[:self.split_lengths["test"]]
        test_set = input_list[self.split_lengths["test"]:]
        return trainval_set, test_set