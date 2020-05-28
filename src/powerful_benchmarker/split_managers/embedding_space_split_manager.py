from .base_split_manager import BaseSplitManager
from .class_disjoint_split_manager import ClassDisjointSplitManager
from ..utils import common_functions as c_f, dataset_utils as d_u
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import numpy as np
import torch

class EmbeddingSpaceSplitManager(ClassDisjointSplitManager):
    # val_set_info: This should be a list of lists. Each sublist consists of [val_size, difficulty]
    def __init__(self, 
            model,
            val_set_info,
            test_size,
            test_difficulty,
            embedding_retrieval_kwargs=None,
            *args, 
            **kwargs
        ):
        BaseSplitManager.__init__(self, *args, **kwargs)
        self.model = model
        self.val_set_info = val_set_info
        self.test_size = test_size
        self.test_difficulty = test_difficulty
        embedding_retrieval_kwargs = {} if embedding_retrieval_kwargs is None else embedding_retrieval_kwargs
        self.tester = testers.GlobalEmbeddingSpaceTester(data_and_label_getter=self.data_and_label_getter, **embedding_retrieval_kwargs)

    def split_assertions(self):
        BaseSplitManager.split_assertions(self)
        self.class_disjoint_assertion()

    def _create_split_schemes(self, datasets):
        # the assumption here is all input datasets are the same
        sample_dataset = datasets["eval"]["train"]
        embeddings, labels = self.tester.get_all_embeddings(sample_dataset, self.model)
        labels = labels[:, 0]
        embeddings = torch.from_numpy(embeddings).to(self.tester.data_device)
        labels = torch.from_numpy(labels).to(self.tester.data_device)

        class_medians = self.get_class_medians(embeddings, labels)
        num_classes = class_medians.size(0)
        num_test = int(num_classes * self.test_size)
        test_set_classes = self.get_classes_by_difficulty(class_medians, num_test, self.test_difficulty == "hard")
        trainval_classes = []
        for (val_size, val_difficulty) in self.val_set_info:
            num_val = int(num_classes * val_size)
            val_set_classes = self.get_classes_by_difficulty(class_medians, num_val, val_difficulty == "hard", classes_already_taken=test_set_classes)
            train_set_classes = list((set(range(num_classes)) - set(test_set_classes)) - set(val_set_classes))
            trainval_classes.append((train_set_classes, val_set_classes))

        return d_u.create_subset_datasets_from_indices(datasets, 
                                                        trainval_classes,
                                                        np.arange(num_classes), 
                                                        test_set_classes, 
                                                        self.get_split_scheme_name,
                                                        len(trainval_classes),
                                                        self.convert_to_subset_idx)


    def get_classes_by_difficulty(self, class_medians, size_of_split, get_hard_subset, classes_already_taken=()):
        filler = float('-inf') if get_hard_subset else float('inf')
        get_next_class = torch.argmax if get_hard_subset else torch.argmin
        subset_processor = (lambda x: x) if get_hard_subset else (lambda x: torch.mean(x, dim=0))

        num_classes = class_medians.size(0)
        sim_mat = lmu.sim_mat(class_medians)
        sim_mat.fill_diagonal_(filler)
        sim_mat[:, classes_already_taken] = filler
        sim_mat[classes_already_taken, :] = filler

        [row, col] = np.unravel_index(get_next_class(sim_mat), (num_classes, num_classes))
        classes = [row, col]
        sim_mat[:, row] = filler

        for i in range(0, size_of_split-2):
            sim_mat[:, col] = filler
            subset = sim_mat[classes, :]
            subset = subset_processor(subset)
            _, col = np.unravel_index(get_next_class(subset), (len(subset), num_classes))
            classes.append(col)

        return classes


    def get_class_medians(self, embeddings, labels):
        label_set = torch.unique(labels)
        medoids = torch.zeros((len(label_set), embeddings.size(1)))
        for i in range(len(label_set)):
            f = embeddings[labels == label_set[i], :]
            medoids[i, :], _ = torch.median(f, dim=0)
        return medoids


    def get_base_split_scheme_name(self):
        return 'Val{}{:02d}_Test{}{:02d}'


    def get_split_scheme_name(self, partition):
        val_size, val_difficulty = self.val_set_info[partition]
        val_size, test_size = int(val_size*100), int(self.test_size*100)
        return self.get_base_split_scheme_name().format(val_difficulty.capitalize(), val_size, 
                                                        self.test_difficulty.capitalize(), test_size)