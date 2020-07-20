from easy_module_attribute_getter import utils as emag_utils

class BaseFactory:
    def __init__(self, getter):
        self.getter = getter
        
    def create(self, specs=None, named_specs=None, subset=None, kwargs=None, per_name_kwargs=None):
        assert (specs is not None) or (named_specs is not None), "Either specs or named_specs has to be set"
        if kwargs is None:
            kwargs = {}
        if named_specs is not None:
            if named_specs == {}:
                return {}
            if per_name_kwargs is None:
                per_name_kwargs = {k:{} for k in named_specs.keys()}
            if subset is None:
                subset = list(named_specs.keys())
            if isinstance(subset, (list, tuple)):
                output = {}
                for k in self.creation_order(named_specs):
                    if k in subset:
                        output[k] = self._create(named_specs, k, kwargs, per_name_kwargs)
                output = self.format_named_specs_output(output)
            else:
                output = self._create(named_specs, subset, kwargs, per_name_kwargs)
            return output
        else:
            if specs == {}:
                return None
            kwargs = self.merge_key_specific_kwargs(kwargs, {}, None)
            return self._create_general(specs, **kwargs)

    def _create(self, named_specs, key, kwargs, per_name_kwargs):
        v = named_specs[key]
        try:
            creator_func = getattr(self, "create_{}".format(key))
        except AttributeError:
            creator_func = self._create_general
        final_kwargs = self.merge_key_specific_kwargs(kwargs, per_name_kwargs[key], key)
        return creator_func(v, **final_kwargs)

    def merge_key_specific_kwargs(self, kwargs, kwargs_for_this_name, key):
        key_specific_kwargs = self.key_specific_kwargs(key)
        x = emag_utils.merge_two_dicts(kwargs, key_specific_kwargs)
        return emag_utils.merge_two_dicts(x, kwargs_for_this_name)

    def key_specific_kwargs(self, key):
        return {}

    def creation_order(self, named_specs):
        return list(named_specs.keys())

    def _create_general(self, specs):
        raise NotImplementedError

    def format_named_specs_output(self, named_specs_output):
        return named_specs_output