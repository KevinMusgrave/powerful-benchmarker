from .base_split_manager import BaseSplitManager
from ..utils import common_functions as c_f
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import numpy as np
import torch

class EmbeddingSpaceSplitManager(BaseSplitManager):
    def __init__(self, 
            model, 
            primary_metric,
            train_size,
            val_size,
            test_size,
            accuracy_calculator,
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert train_size + val_size + test_size == 1
        self.model = model
        self.primary_metric = primary_metric
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        self.tester = testers.GlobalEmbeddingSpaceTester(dataloader_num_workers=2, data_and_label_getter= lambda data: (data["data"], data["label"]))
        self.accuracy_calculator = accuracy_calculator

    def _create_split_schemes(self, datasets):
        # sample_dataset = c_f.first_val_of_dict(c_f.first_val_of_dict(datasets))
        sample_dataset = datasets["eval"]["val"]
        embeddings, labels = self.tester.get_all_embeddings(sample_dataset, self.model)
        labels = labels[:, 0]
        embeddings = torch.from_numpy(embeddings).to(self.tester.data_device)
        labels = torch.from_numpy(labels).to(self.tester.data_device)
        class_medians = self.get_class_medians(embeddings, labels)
        num_C = class_medians.size(0)
        num_train, num_val = [int(num_C*x) for x in [self.train_size, self.val_size]]
        num_test = int(num_C - (num_train + num_val))

        sim_mat = lmu.sim_mat(class_medians)
        sim_mat.fill_diagonal_(float('-inf'))

        [row, col] = np.unravel_index(torch.argmax(sim_mat), (num_C, num_C))
        test_classes = [row, col]
        sim_mat[:, row] = float('-inf')

        for i in range(0, num_test-2):
            sim_mat[:, col] = float('-inf')
            subset = sim_mat[test_classes, :]
            subset = torch.mean(subset, dim=0)
            _, col = np.unravel_index(torch.argmax(subset), (len(subset), num_C))
            test_classes.append(col)

        # test_classes = list(range(100,200))

        print("test_classes", test_classes)
        embeddings, labels = embeddings.cpu().numpy(), labels.cpu().numpy()
        test_set_indices = np.where(np.isin(labels, test_classes))[0]
        test_set_embeddings = embeddings[test_set_indices]
        test_set_labels = labels[test_set_indices]
        print("test_set size=", test_set_embeddings.shape)
        print("test_set_labels size=", test_set_labels.shape)
        print("num classes=", len(test_classes))
        print("num unique classes=", len(set(test_classes)))
        accuracies = self.accuracy_calculator.get_accuracy(test_set_embeddings,test_set_embeddings,test_set_labels,test_set_labels,True) 
        print(accuracies)


    def get_class_medians(self, embeddings, labels):
        label_set = torch.unique(labels)
        medoids = torch.zeros((len(label_set), embeddings.size(1)))
        for i in range(len(label_set)):
            f = embeddings[labels == label_set[i], :]
            medoids[i, :], _ = torch.median(f, dim=0)
        return medoids