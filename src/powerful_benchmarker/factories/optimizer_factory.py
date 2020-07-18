from ..utils import common_functions as c_f
from .base_factory import BaseFactory
import logging
import copy

class OptimizerFactory(BaseFactory):
    def __init__(self, param_sources=None, **kwargs):
        super().__init__(**kwargs)
        self.param_sources = param_sources

    def _create_general(self, optimizer_type, k):
        basename = k.replace("_optimizer", '')
        param_source = None
        for possible_params in self.param_sources:
            if basename in possible_params:
                param_source = possible_params[basename]
                break
        assert param_source is not None, "A matching parameter source could not be found for {}".format(k)
        o, s, g = self.getter.get_optimizer(param_source, yaml_dict=optimizer_type)
        output = {}
        logging.info("%s\n%s" % (k, o))
        if o is not None: output["optimizer"] = o
        if s is not None: output["lr_scheduler"] = {"%s_%s"%(basename, x):v for x,v in s.items()}
        if g is not None: output["gradient_clipper"] = g
        return output


    def key_specific_kwargs(self, key):
        return {"k": key}

    def format_named_specs_output(self, named_specs_output):
        optimizers, lr_schedulers, gradient_clippers = {}, {}, {}
        for k,v in named_specs_output.items():
            if "optimizer" in v: optimizers[k] = v["optimizer"]
            if "lr_scheduler" in v: lr_schedulers[k] = v["lr_scheduler"]
            if "gradient_clipper" in v: gradient_clipper[k] = v["gradient_clipper"]
        return optimizers, lr_schedulers, gradient_clippers