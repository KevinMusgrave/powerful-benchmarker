#! /usr/bin/env python3

from .. import architectures as arch
from .api_train_with_classifier import APIMaybeExtendTrainWithClassifier


class APIDeepAdversarialMetricLearning(APIMaybeExtendTrainWithClassifier):
    def get_trainer_kwargs(self):
        trainer_kwargs = self.inheriter.get_trainer_kwargs()
        trainer_kwargs["g_alone_epochs"] = self.args.g_alone_epochs
        trainer_kwargs["metric_alone_epochs"] = self.args.metric_alone_epochs
        trainer_kwargs["g_triplets_per_anchor"] = self.args.g_triplets_per_anchor
        return trainer_kwargs

    def set_model(self):
        self.inheriter.set_model()
        self.models["generator"] = arch.misc_models.MLP(
            [self.base_model_output_size * 3,
             self.base_model_output_size,
             self.base_model_output_size], 
            final_relu=True
        )

