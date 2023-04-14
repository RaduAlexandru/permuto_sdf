#https://github.com/devforfu/pytorch_playground/blob/master/loop.ipynb

import re
import torch

def to_snake_case(string):
    """Converts CamelCase string into snake_case."""
    
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def classname(obj):
    return obj.__class__.__name__


class Callback:
    """
    The base class inherited by callbacks.

    Provides a lot of hooks invoked on various stages of the training loop
    execution. The signature of functions is as broad as possible to allow
    flexibility and customization in descendant classes.
    """
    def training_started(self, **kwargs): pass

    def training_ended(self, **kwargs): pass

    def epoch_started(self, **kwargs): pass

    def phase_started(self, **kwargs): pass

    def phase_ended(self, **kwargs): pass

    def epoch_ended(self, **kwargs): pass

    def batch_started(self, **kwargs): pass

    def batch_ended(self, **kwargs): pass

    def before_forward_pass(self, **kwargs): pass

    def after_forward_pass(self, **kwargs): pass

    def before_backward_pass(self, **kwargs): pass

    def after_backward_pass(self, **kwargs): pass


class CallbacksGroup(Callback):
    """
    Groups together several callbacks and delegates training loop 
    notifications to the encapsulated objects.
    """
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.named_callbacks = {to_snake_case(classname(cb)): cb for cb in self.callbacks}
        
    def __getitem__(self, item):
        item = to_snake_case(item)
        if item in self.named_callbacks:
            return self.named_callbacks[item]
        raise KeyError(f'callback name is not found: {item}')
        
    def training_started(self, **kwargs): self.invoke('training_started', **kwargs)

    def training_ended(self, **kwargs): self.invoke('training_ended', **kwargs)

    def epoch_started(self, **kwargs): self.invoke('epoch_started', **kwargs)

    def phase_started(self, **kwargs): self.invoke('phase_started', **kwargs)

    def phase_ended(self, **kwargs): self.invoke('phase_ended', **kwargs)

    def epoch_ended(self, **kwargs): self.invoke('epoch_ended', **kwargs)

    def batch_started(self, **kwargs): self.invoke('batch_started', **kwargs)

    def batch_ended(self, **kwargs): self.invoke('batch_ended', **kwargs)

    def before_forward_pass(self, **kwargs): self.invoke('before_forward_pass', **kwargs)

    def after_forward_pass(self, **kwargs): self.invoke('after_forward_pass', **kwargs)

    def before_backward_pass(self, **kwargs): self.invoke('before_backward_pass', **kwargs)

    def after_backward_pass(self, **kwargs): self.invoke('after_backward_pass', **kwargs)
    
    def invoke(self, method, **kwargs):
        with torch.set_grad_enabled(False):
            for cb in self.callbacks:
                getattr(cb, method)(**kwargs)