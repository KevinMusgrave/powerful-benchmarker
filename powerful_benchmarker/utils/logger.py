from pytorch_adapt.frameworks.ignite import IgniteValHookWrapper
from pytorch_adapt.frameworks.ignite.loggers import (
    BasicLossLogger,
    IgniteRecordKeeperLogger,
)
from pytorch_adapt.utils import common_functions as c_f


class Logger:
    def __init__(self, folder, record_keeper_freq=50):
        self.logger1 = BasicLossLogger()
        self.logger2 = IgniteRecordKeeperLogger(folder=folder)
        self.record_keeper_freq = record_keeper_freq

    def add_training(self, adapter):
        fn1 = self.logger1.add_training(adapter)
        fn2 = self.logger2.add_training(adapter)

        def fn(engine):
            fn1(engine)
            if engine.state.iteration % self.record_keeper_freq == 0:
                fn2(engine)

        return fn

    def add_validation(self, *args, **kwargs):
        self.logger2.add_validation(*args, **kwargs)

    def write(self, *args, **kwargs):
        self.logger2.write(*args, **kwargs)

    def get_losses(self):
        return self.logger1.get_losses()


class IgniteValHookWrapperWithPrint(IgniteValHookWrapper):
    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        c_f.LOGGER.info(self.validator)
