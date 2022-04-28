import numpy as np

from .main_utils import domain_len_assertion


def pretrained_src_accuracy(dataset, src_domains, split, average):
    src_domain = domain_len_assertion(src_domains)

    mnist = {
        "mnist": {
            "train": {"micro": 0.9994333386421204, "macro": 0.9994416236877441},
            "val": {"micro": 0.9951000213623047, "macro": 0.9949950575828552},
        }
    }

    office31 = {
        "amazon": {
            "train": {"micro": 0.9880159497261047, "macro": 0.9868927001953125},
            "val": {"micro": 0.9042553305625916, "macro": 0.9006038904190063},
        },
        "dslr": {
            "train": {"micro": 0.9974874258041382, "macro": 0.9946236610412598},
            "val": {"micro": 0.9900000095367432, "macro": 0.9892473220825195},
        },
        "webcam": {
            "train": {"micro": 1.0, "macro": 1.0},
            "val": {"micro": 0.9811320900917053, "macro": 0.9784946441650391},
        },
    }

    officehome = {
        "art": {
            "train": {"micro": 0.9953632354736328, "macro": 0.99530029296875},
            "val": {"micro": 0.8148148059844971, "macro": 0.7875152826309204},
        },
        "clipart": {
            "train": {"micro": 0.955899178981781, "macro": 0.9570373892784119},
            "val": {"micro": 0.7537227869033813, "macro": 0.7572566270828247},
        },
        "product": {
            "train": {"micro": 0.9909884333610535, "macro": 0.99105304479599},
            "val": {"micro": 0.9268018007278442, "macro": 0.9206826686859131},
        },
        "real": {
            "train": {"micro": 0.9845049977302551, "macro": 0.980871319770813},
            "val": {"micro": 0.8841742873191833, "macro": 0.868537187576294},
        },
    }

    x = {"mnist": mnist, "office31": office31, "officehome": officehome}

    return np.round(x[dataset][src_domain][split][average], 4)


def pretrained_target_accuracy(dataset, src_domains, target_domains):
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
