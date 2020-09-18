from .base_factory import BaseFactory
from easy_module_attribute_getter import utils as emag_utils
import copy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import logging

class TesterFactory(BaseFactory):
    def _create_general(self, tester_type, plots_folder, **kwargs):
        tester, tester_params = self.getter.get("tester", yaml_dict=tester_type, return_uninitialized=True)
        tester_params = copy.deepcopy(tester_params)
        tester_params["accuracy_calculator"] = self.getter.get("accuracy_calculator", yaml_dict=tester_params["accuracy_calculator"])
        if tester_params.get("visualizer", None):
            tester_params["visualizer"] = self.getter.get("visualizer", yaml_dict=tester_params["visualizer"])
            tester_params["visualizer_hook"] = self.visualizer_hook(plots_folder)
        tester_params = emag_utils.merge_two_dicts(tester_params, kwargs)
        return tester(**tester_params)

    def visualizer_hook(self, plots_folder):
        def actual_visualizer_hook(visualizer, embeddings, labels, split_name, keyname, epoch):
            logging.info("Plot for the {} split and label set {}".format(split_name, keyname))
            label_set = np.unique(labels)
            num_classes = len(label_set)
            fig = plt.figure(figsize=(16,12))
            plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
            for i in range(num_classes):
                idx = labels == label_set[i]
                plt.plot(embeddings[idx, 0], embeddings[idx, 1], ".", markersize=1)   
            plt.savefig(os.path.join(plots_folder, "{}_epoch{}.png".format(keyname, epoch)))
        return actual_visualizer_hook
               