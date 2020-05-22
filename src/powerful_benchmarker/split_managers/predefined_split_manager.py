from .base_split_manager import BaseSplitManager

class PredefinedSplitManager(BaseSplitManager):
    def _create_split_schemes(self, datasets):
        return {self.get_split_scheme_name(0): datasets}

    def split_assertions(self):
        pass

    def get_base_split_scheme_name(self):
        return "PredefinedSplitScheme_"

    def get_split_scheme_name(self, partition):
        return "{}{}".format(self.get_base_split_scheme_name(), partition)