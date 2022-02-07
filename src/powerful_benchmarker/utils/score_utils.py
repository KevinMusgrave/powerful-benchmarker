import numpy as np

from .main_utils import domain_len_assertion


def pretrained_src_val_accuracy(dataset, src_domains):
    src_domain = domain_len_assertion(src_domains)

    x = {
        "mnist": {"mnist": 0.9949950575828552},
        "office31": {
            "amazon": 0.9006038904190063,
            "dslr": 0.9892473220825195,
            "webcam": 0.9784946441650391,
        },
        "officehome": {
            "art": 0.7875152826309204,
            "clipart": 0.7572566270828247,
            "product": 0.9206826686859131,
            "real": 0.868537187576294,
        },
    }
    return np.round(x[dataset][src_domain], 4)


def pretrained_target_train_accuracy(dataset, src_domains, target_domains):
    src_domain = domain_len_assertion(src_domains)
    target_domain = domain_len_assertion(target_domains)

    x = {
        "mnist": {"mnist": {"mnistm": 0.5797358751296997}},
        "office31": {
            "amazon": {"dslr": 0.8512204885482788, "webcam": 0.8189197778701782},
            "dslr": {"amazon": 0.6932609677314758, "webcam": 0.941096305847168},
            "webcam": {"amazon": 0.713221549987793, "dslr": 0.9890364408493042},
        },
        "officehome": {
            "art": {
                "clipart": 0.41807034611701965,
                "product": 0.6699330806732178,
                "real": 0.7531952857971191,
            },
            "clipart": {
                "art": 0.5499463081359863,
                "product": 0.6609126925468445,
                "real": 0.6891355514526367,
            },
            "product": {
                "art": 0.5802421569824219,
                "clipart": 0.41485580801963806,
                "real": 0.750948429107666,
            },
            "real": {
                "art": 0.6575398445129395,
                "clipart": 0.44673898816108704,
                "product": 0.7783608436584473,
            },
        },
    }
    return np.round(x[dataset][src_domain][target_domain], 4)
