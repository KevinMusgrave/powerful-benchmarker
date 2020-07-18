from easy_module_attribute_getter import utils as emag_utils

class BaseFactory:
    def __init__(self, getter):
        self.getter = getter
        
    def create(self, specs=None, named_specs=None, subset=None, additional_kwargs=None):
        assert (specs is not None) or (named_specs is not None), "Either specs or named_specs has to be set"
        if named_specs is not None:
            if named_specs == {}:
                return {}
            if additional_kwargs is None:
                additional_kwargs = {k:{} for k in named_specs.keys()}
            if subset is None:
                subset = list(named_specs.keys())
            if isinstance(subset, (list, tuple)):
                output = {}
                for k in self.creation_order(named_specs):
                    if k in subset:
                        output[k] = self._create(named_specs, k, additional_kwargs)
            else:
                output = self._create(named_specs, subset, additional_kwargs)
            return output
        else:
            if specs == {}:
                return None
            if additional_kwargs is None:
                additional_kwargs = {}
            kwargs = self.maybe_add_more_kwargs(additional_kwargs, "general")
            return self._create_general(specs, **kwargs)

    def _create(self, named_specs, key, additional_kwargs):
        v = named_specs[key]
        try:
            creator_func = getattr(self, "create_{}".format(key))
        except AttributeError:
            creator_func = self._create_general
        kwargs = self.maybe_add_more_kwargs(additional_kwargs[key], key)
        return creator_func(v, **kwargs)

    def maybe_add_more_kwargs(self, additional_kwargs, name):
        try:
            even_more_kwargs = getattr(self, "{}_kwargs".format(name))()
            return emag_utils.merge_two_dicts(additional_kwargs, even_more_kwargs)
        except AttributeError:
            return additional_kwargs

    def creation_order(self, named_specs):
        return list(named_specs.keys())

    def _create_general(self, specs):
        raise NotImplementedError