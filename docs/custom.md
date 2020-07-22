# Adding Custom Modules

### Register your own classes and modules
By default, the API gives you access to losses/miners/datasets/optimizers/schedulers/trainers etc that are available in powerful-benchmarker, PyTorch, and pytorch-metric-learning.

Let's say you make your own loss and mining functions, and you'd like to have access to them via the API. You can accomplish this by replacing the last two lines of the [example script](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/examples/run.py) with this:

```python
from pytorch_metric_learning import losses, miners

# your custom loss function
class YourLossFunction(losses.BaseMetricLossFunction):
   ...

# your custom mining function
class YourMiningFunction(miners.BaseTupleMiner):
   ...

r = runner(**(args.__dict__))

# make the runner aware of them
r.register("loss", YourLossFunction)
r.register("miner", YourMiningFunction)
r.run()
```

Now you can access your custom classes just like any other class:
```yaml
loss_funcs:
  metric_loss: 
    YourLossFunction:

mining_funcs:
  tuple_miner:
    YourMiningFunction:
```

If you have a module containing multiple classes and you want to register all those classes, you can simply register the module:
```python
import YourModuleOfLosses
r.register("loss", YourModuleOfLosses)
```

Registering your own trainer is a bit more involved, because you need to also create an associated API parser. The name of the api parser should be ```APIParser<name of your training method>```. 

Here's an example where I make a trainer that extends ```trainers.MetricLossOnly```, and takes in an additional argument ```foo```. In order to pass this in, the API parser needs to add ```foo``` to the trainer kwargs, and this is done in the ```get_trainer_kwargs``` method.

```python
from pytorch_metric_learning import trainers
from powerful_benchmarker import api_parsers

class YourTrainer(trainers.MetricLossOnly):
    def __init__(self, foo, **kwargs):
	super().__init__(**kwargs)
	self.foo = foo
	print("foo = ", self.foo)


class APIYourTrainer(api_parsers.BaseAPIParser):
    def get_foo(self):
        return "hello"

    def get_trainer_kwargs(self):
        trainer_kwargs = super().get_trainer_kwargs()
        trainer_kwargs["foo"] = self.get_foo()
        return trainer_kwargs

r = runner(**(args.__dict__))
r.register("trainer", YourTrainer)
r.register("api_parser", APIYourTrainer)
r.run()
```