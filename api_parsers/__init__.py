#! /usr/bin/env python3

from .api_cascaded_embeddings import APICascadedEmbeddings
from .api_deep_adversarial_metric_learning import APIParserDeepAdversarialMetricLearning
from .api_train_with_classifier import APIParserTrainWithClassifier
from .base_api_parser import BaseAPIParser


api_parsers_dict = {
    "MetricLossOnly": BaseAPIParser,
    "DeepAdversarialMetricLearning": APIParserDeepAdversarialMetricLearning,
    "TrainWithClassifier": APIParserTrainWithClassifier,
    "CascadedEmbeddings": APICascadedEmbeddings,
}
