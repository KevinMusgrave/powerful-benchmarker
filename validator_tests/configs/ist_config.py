from pytorch_adapt.validators import ISTValidator

from .base_config import BaseConfig, get_full_split_name, use_src_and_target


class IST(BaseConfig):
    def __init__(self, config):
        super().__init__(config)
        self.validator_args["with_ent"] = bool(int(self.validator_args["with_ent"]))
        self.validator_args["with_div"] = bool(int(self.validator_args["with_div"]))
        self.layer = self.validator_args["layer"]
        self.src_split_name = get_full_split_name("src", self.split)
        self.target_split_name = get_full_split_name("target", self.split)
        self.validator = ISTValidator(
            key_map={
                self.src_split_name: "src_train",
                self.target_split_name: "target_train",
            },
            batch_size=512,
            layer=self.validator_args["layer"],
            with_ent=self.validator_args["with_ent"],
            with_div=self.validator_args["with_div"],
        )

    def score(self, x, exp_config, device):
        return use_src_and_target(
            x,
            device,
            self.validator,
            self.src_split_name,
            self.target_split_name,
            self.layer,
        )

    def expected_keys(self):
        return {"with_ent", "with_div", "layer", "split"}
