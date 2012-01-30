import os
modules = []
for name in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    m, ext = os.path.splitext(name)
    if ext == '.py' and m !='__init__':
        __import__(__name__+"."+m)
        modules.append(m)

__all__ = modules
