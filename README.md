# SentenceGraph

This tiny module allows you to parse text with UDPipe models and manage parsed sentences. See **Guide.ipynb** for details.

# Installation

```bash
$ python setup.py install
```

In order to be able to plot graphs you need to install next packages:
```
$ sudo apt install python-pydot python-pydot-ng graphviz
```
# Where to get a UDPipe model

Currenly, the Russian UDpipe model is provided. You can train your own model for other languages. English model, in particular, can be found [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2998). The guide how to train your own model see [here](https://astromis.github.io/2019/05/15/discover_the_udpipe.html). 

# Relateg projects:

* [UDPipe](https://github.com/ufal/udpipe)
* [Finnish-dep-parser](https://github.com/TurkuNLP/Finnish-dep-parser)
* [CoNNLU site](https://universaldependencies.org/format.html)
