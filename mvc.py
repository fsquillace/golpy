"""
MVC Design Pattern 

@author: Filippo Squillace
@date: 03/12/2011
"""


class Observer:
    def update(*args, **kwargs):
        raise NotImplementedError


class Observable:
    def __init__(self):
        self.observers = []
        pass

    def register(self, observer):
        self.observers.append(observer)

    def notify(self, *args, **kwargs):
        for obs in self.observers:
            obs.update(args, kwargs)
