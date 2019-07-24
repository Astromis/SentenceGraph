# SubsentenceSplitter

This program allows you to split the complex sentence on its simpler parts. For example

Source:
```
I can write this sentence and split it on the parts.
```
Output:
```
I can write this sentence
and split it on the parts.
```
It based on UDpipe syntax parser so an error will be depends on the accuracy of parsing. Here the only Russian model is available due to space limitations, but actually, for other languages, English in particular, you can find the model [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2998) or train you own model. How to do this, see [here](https://astromis.github.io/2019/05/15/discover_the_udpipe.html) 


Code sample:

```python
from udpipe_model import Model
from splitter import subsent_tokenze
text = '''я пополнил сегодня телефон на 430 рублей, на счёт поступило 260 и пришла смс, что подписка на эплмюзик возобновлена.'''

model = Model('./models/rus_model.udpipe')
subsent_tokenze(text, model)
```
Relateg projects:

[UDPipe](https://github.com/ufal/udpipe)
[Finnish-dep-parser](https://github.com/TurkuNLP/Finnish-dep-parser)
