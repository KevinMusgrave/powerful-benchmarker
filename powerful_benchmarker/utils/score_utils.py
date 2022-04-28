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


def pretrained_target_accuracy(dataset, src_domains, target_domains, split, average):
    src_domain = domain_len_assertion(src_domains)
    target_domain = domain_len_assertion(target_domains)

    mnist = {
        "mnist": {
            "mnistm": {
                "train": {"micro": 0.576329231262207, "macro": 0.5797358751296997},
                "val": {"micro": 0.5739362239837646, "macro": 0.5770031213760376},
            }
        }
    }

    office31 = {
        "amazon": {
            "dslr": {
                "train": {"micro": 0.8266331553459167, "macro": 0.8512204885482788},
                "val": {"micro": 0.7799999713897705, "macro": 0.7833333015441895},
            },
            "webcam": {
                "train": {"micro": 0.805031418800354, "macro": 0.8189197778701782},
                "val": {"micro": 0.7735849022865295, "macro": 0.7740142941474915},
            },
        },
        "dslr": {
            "amazon": {
                "train": {"micro": 0.6950732469558716, "macro": 0.6932609677314758},
                "val": {"micro": 0.695035457611084, "macro": 0.6928660869598389},
            },
            "webcam": {
                "train": {"micro": 0.9418238997459412, "macro": 0.941096305847168},
                "val": {"micro": 0.9245283007621765, "macro": 0.9129031896591187},
            },
        },
        "webcam": {
            "amazon": {
                "train": {"micro": 0.7150465846061707, "macro": 0.713221549987793},
                "val": {"micro": 0.7269503474235535, "macro": 0.7322812676429749},
            },
            "dslr": {
                "train": {"micro": 0.9899497628211975, "macro": 0.9890364408493042},
                "val": {"micro": 0.9800000190734863, "macro": 0.9811828136444092},
            },
        },
    }

    officehome = {
        "art": {
            "clipart": {
                "train": {"micro": 0.41151201725006104, "macro": 0.41807034611701965},
                "val": {"micro": 0.4192439913749695, "macro": 0.4325830340385437},
            },
            "product": {
                "train": {"micro": 0.6860039234161377, "macro": 0.6699330806732178},
                "val": {"micro": 0.7038288116455078, "macro": 0.6905300617218018},
            },
            "real": {
                "train": {"micro": 0.7667145133018494, "macro": 0.7531952857971191},
                "val": {"micro": 0.7729358077049255, "macro": 0.7549898624420166},
            },
        },
        "clipart": {
            "art": {
                "train": {"micro": 0.6017516851425171, "macro": 0.5499463081359863},
                "val": {"micro": 0.6193415522575378, "macro": 0.5711568593978882},
            },
            "product": {
                "train": {"micro": 0.6764291524887085, "macro": 0.6609126925468445},
                "val": {"micro": 0.6869369149208069, "macro": 0.6790676712989807},
            },
            "real": {
                "train": {"micro": 0.7047345638275146, "macro": 0.6891355514526367},
                "val": {"micro": 0.6961008906364441, "macro": 0.6748995184898376},
            },
        },
        "product": {
            "art": {
                "train": {"micro": 0.6002060770988464, "macro": 0.5802421569824219},
                "val": {"micro": 0.604938268661499, "macro": 0.5949786305427551},
            },
            "clipart": {
                "train": {"micro": 0.428121417760849, "macro": 0.41485580801963806},
                "val": {"micro": 0.4249713718891144, "macro": 0.4174821376800537},
            },
            "real": {
                "train": {"micro": 0.7618364691734314, "macro": 0.750948429107666},
                "val": {"micro": 0.7855504751205444, "macro": 0.7742912173271179},
            },
        },
        "real": {
            "art": {
                "train": {"micro": 0.687789797782898, "macro": 0.6575398445129395},
                "val": {"micro": 0.7037037014961243, "macro": 0.6948687434196472},
            },
            "clipart": {
                "train": {"micro": 0.4470217525959015, "macro": 0.44673898816108704},
                "val": {"micro": 0.4387170672416687, "macro": 0.450015664100647},
            },
            "product": {
                "train": {"micro": 0.7907631397247314, "macro": 0.7783608436584473},
                "val": {"micro": 0.7860360145568848, "macro": 0.7748997211456299},
            },
        },
    }

    x = {"mnist": mnist, "office31": office31, "officehome": officehome}
    return np.round(x[dataset][src_domain][target_domain][split][average], 4)
