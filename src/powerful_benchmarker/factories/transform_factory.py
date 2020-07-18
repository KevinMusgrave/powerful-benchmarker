from .base_factory import BaseFactory
from .model_factory import ModelFactory

class TransformFactory(BaseFactory):
    def _create_general(self, transform_type):
        try:
            model_factory = ModelFactory(api_parser=self.api_parser, getter=self.getter)
            trunk_model = model_factory.create(named_specs=self.api_parser.args.models, subset="trunk")
            model_transform_properties = {k:getattr(trunk_model, k) for k in ["mean", "std", "input_space", "input_range"]}
        except (KeyError, AttributeError):
            model_transform_properties = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        return self.getter.get_composed_img_transform(transform_type, **model_transform_properties)