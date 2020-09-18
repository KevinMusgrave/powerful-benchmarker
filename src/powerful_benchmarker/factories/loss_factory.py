from pytorch_metric_learning import losses
import copy
from .base_factory import BaseFactory
from ..utils import common_functions as c_f
from pytorch_metric_learning.utils.common_functions import TorchInitWrapper
import logging

class LossFactory(BaseFactory):
    nested_regularizer_objects = ["distance", "reducer"]
    nested_loss_objects = nested_regularizer_objects + ["weight_regularizer", "embedding_regularizer", "weight_init_func"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from .miner_factory import MinerFactory
        self.miner_factory = MinerFactory

    def _create_general(self, loss_type, num_classes):
        loss, loss_params = self.getter.get("loss", yaml_dict=loss_type, return_uninitialized=True)
        loss_params = copy.deepcopy(loss_params)
        if loss == losses.MultipleLosses:
            loss_funcs = [self.create({k:v}) for k,v in loss_params["losses"].items()]
            return loss(loss_funcs) 

        if loss == losses.CrossBatchMemory:
            if "loss" in loss_params: 
                loss_params["loss"] = self.create(loss_params["loss"])
            if "miner" in loss_params: 
                self.miner_factory = self.miner_factory(getter=self.getter)
                loss_params["miner"] = self.miner_factory.create(loss_params["miner"])

        if c_f.check_init_arguments(loss, "num_classes") and ("num_classes" not in loss_params):
            loss_params["num_classes"] = num_classes
            logging.info("Passing %d as num_classes to the loss function"%loss_params["num_classes"])

        for obj_name in self.nested_loss_objects:
            if obj_name in loss_params:
                if "regularizer" in obj_name: 
                    loss_params[obj_name] = self.get_nested_regularizer(loss_params[obj_name])
                elif obj_name == "weight_init_func":
                    loss_params[obj_name] = self.get_weight_init_func(loss_params[obj_name])
                else:
                    loss_params[obj_name] = self.getter.get(obj_name, yaml_dict=loss_params[obj_name])

        return loss(**loss_params)


    def get_nested_regularizer(self, regularizer_type):
        regularizer, regularizer_params = self.getter.get("regularizer", yaml_dict=regularizer_type, return_uninitialized=True)
        for obj_name in self.nested_regularizer_objects:
            if obj_name in regularizer_params:
                regularizer_params[obj_name] = self.getter.get(obj_name, yaml_dict=regularizer_params[obj_name])
        return regularizer(**regularizer_params)


    def get_weight_init_func(self, weight_init_type):
        weight_init, weight_init_params = self.getter.get("weight_init_func", yaml_dict=weight_init_type, return_uninitialized=True)
        return TorchInitWrapper(weight_init, **weight_init_params)