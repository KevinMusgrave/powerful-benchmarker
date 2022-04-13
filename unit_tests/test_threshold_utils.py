import unittest

import numpy as np
import pandas as pd

from validator_tests.utils.threshold_utils import (
    TARGET_ACCURACY,
    get_avg_top_n_acc_by_group,
    group_by_task,
    group_by_task_validator,
)


class TestThresholdUtils(unittest.TestCase):
    def test_get_avg_top_n_acc_by_group(self):
        validators = ["A", "B", "C"]
        validator_args_list = ["x", "y", "z"]
        adapters = ["DANN", "MCD", "MMD"]
        size = 100

        df = {
            "validator": np.random.choice(validators, size=(size,)),
            "validator_args": np.random.choice(validator_args_list, size=(size,)),
            "dataset": ["mnist"] * size,
            "src_domains": [("mnist",)] * size,
            "target_domains": [("mnistm",)] * size,
        }
        df = pd.DataFrame.from_dict(df)
        dfs = []
        for adapter in adapters:
            dfs.append(df.copy().assign(adapter=adapter))
        df = pd.concat(dfs)

        df = df.assign(
            **{
                "score": np.random.randn(len(df)),
                TARGET_ACCURACY: np.random.randn(len(df)),
            }
        )

        # group_by_task(False)
        for nlargest in [1, 2, 3, 4]:
            x = get_avg_top_n_acc_by_group(
                df, group_by_task(False), nlargest, TARGET_ACCURACY, "best_acc"
            )
            direct_compute = df.nlargest(nlargest, TARGET_ACCURACY)[
                TARGET_ACCURACY
            ].mean()
            self.assertTrue(np.isclose(x["best_acc"].item(), direct_compute))

        # group_by_task(True)
        for nlargest in [1, 2, 3, 4]:
            x = get_avg_top_n_acc_by_group(
                df, group_by_task(True), nlargest, TARGET_ACCURACY, "best_acc"
            )

            correct_means = {}
            for adapter in adapters:
                curr_df = df[df["adapter"] == adapter]
                curr_df = curr_df.sort_values([TARGET_ACCURACY], ascending=False)
                correct_means[adapter] = curr_df.head(nlargest)[TARGET_ACCURACY].mean()

            for k, v in correct_means.items():
                self.assertTrue(np.isclose(x[x["adapter"] == k]["best_acc"].item(), v))

        # group_by_task_validator(False)
        for nlargest in [1, 2, 3, 4]:
            x = get_avg_top_n_acc_by_group(
                df,
                group_by_task_validator(False),
                nlargest,
                TARGET_ACCURACY,
                "best_acc",
            )

            correct_means = {}
            for validator in validators:
                for validator_args in validator_args_list:
                    curr_df = df[
                        (df["validator"] == validator)
                        & (df["validator_args"] == validator_args)
                    ]
                    curr_df = curr_df.sort_values([TARGET_ACCURACY], ascending=False)
                    correct_means[f"{validator}.{validator_args}"] = curr_df.head(
                        nlargest
                    )[TARGET_ACCURACY].mean()

            for k, v in correct_means.items():
                validator, validator_args = k.split(".")
                self.assertTrue(
                    np.isclose(
                        x[
                            (x["validator"] == validator)
                            & (x["validator_args"] == validator_args)
                        ]["best_acc"].item(),
                        v,
                    )
                )
