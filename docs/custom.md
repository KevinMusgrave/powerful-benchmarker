# Adding Custom Modules

### Register your own classes and modules
By default, this library gives you access to [various classes in pytorch-metric-learning, torch, torchvision, and pretrainedmodels](index.md#modules-that-can-be-benchmarked).

Let's say you want to use your own loss function as well as a custom optimizer that isn't available in torch.optim. You can accomplish this by replacing the last two lines of the [example script](https://github.com/KevinMusgrave/powerful-benchmarker/blob/master/examples/run.py) with this:

```python
from your_own_loss import YourLossFunction
from custom_optimizer import CoolOptimizer

r = runner(**(args.__dict__))

# make the runner aware of them
r.register("loss", YourLossFunction)
r.register("optimizer", CoolOptimizer)
r.run()
```

Now you can access your custom classes just like any other class:
```yaml
loss_funcs:
  metric_loss: 
    YourLossFunction:

optimizers:
  trunk_optimizer:
    CoolOptimizer:
      lr: 0.01
```

If you have a module containing multiple classes and you want to register all those classes, you can simply register the module:
```python
import YourModuleOfLosses
r.register("loss", YourModuleOfLosses)
```

Registering your own trainer is a bit more involved, because you need to also create an associated API parser. The name of the api parser should be ```APIParser<name of your training method>```.

Here's an example where I make a trainer that extends ```trainers.MetricLossOnly```, and takes in an additional argument ```foo```. If ```foo``` is a simple parameter that can be specified directly in a config file, then ```APIYourTrainer``` doesn't need to do anything other than exist:

```python
from pytorch_metric_learning import trainers
from powerful_benchmarker import api_parsers

class YourTrainer(trainers.MetricLossOnly):
    def __init__(self, foo, **kwargs):
        super().__init__(**kwargs)
        self.foo = foo
        print("foo = ", self.foo)

class APIYourTrainer(api_parsers.BaseAPIParser):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

r = runner(**(args.__dict__))
r.register("trainer", YourTrainer)
r.register("api_parser", APIYourTrainer)
r.run()
```

However, if ```foo``` is more complex, e.g. it is an object that requires some logic to be created, then you'll want ```APIYourTrainer``` to handle that logic, and then add ```foo``` to the ```default_kwargs_trainer``` dictionary. Check out the [code documentation](../code/api_parsers) for details on this.