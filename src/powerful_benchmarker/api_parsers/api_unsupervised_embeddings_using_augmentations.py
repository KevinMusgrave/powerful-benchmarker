from .base_api_parser import BaseAPIParser

class APIUnsupervisedEmbeddingsUsingAugmentations(BaseAPIParser):
    def default_kwargs_trainer(self):
        trainer_kwargs = super().default_kwargs_trainer()
        transforms = self.get_transforms()
        trainer_kwargs["transforms"] = lambda: [transforms[k] for k in transforms.keys() if k.startswith("augmentation")]
        trainer_kwargs["sampler"] = lambda: None
        trainer_kwargs["set_min_label_to_zero"] = lambda: False
        trainer_kwargs["data_and_label_setter"] = lambda: (lambda x: {"data":x[0], "label":x[1]})
        return trainer_kwargs


    def get_transforms(self, **kwargs):
        transforms = super().get_transforms(**kwargs)
        transforms["train"] = None
        return transforms