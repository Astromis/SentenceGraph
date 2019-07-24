from WordGraph import link, WordGraph
from conllu import parse
from udpipe_model import Model


def subsent_tokenze(string, udpipe_model):
    '''
        Build WordGraph sructure from list of strings.
    '''
    print("Loading UDpipe model")
    conllu_sentences = []
    all_subsentences = []
    sentences = model.tokenize(string.lower())
    # Getting dependancy trees
    for s in sentences:
        udpipe_model.tag(s)
        udpipe_model.parse(s)
    conllu = udpipe_model.write(sentences, "conllu")
    try:
        conllu_sentences.append(parse(conllu))
    except:
        return -1
    # Building graph
    for sentences in conllu_sentences:
        for sent in sentences:
            vericlies = []
            for i in sent:
                morph = ''
                if i['feats'] != None:
                    morph = "|".join([k+'='+ v for k,v in i['feats'].items() ])
                vericlies.append( link( i['id'], i['head'], i['lemma'], i['upostag'], i['deprel'], i['form'], morph) )
            all_subsentences.extend(WordGraph(vericlies))
    return all_subsentences


#Example
#text = '''я пополнил сегодня телефон на 430 рублей, на счёт поступило 260 и пришла смс, что подписка на эплмюзик возобновлена.'''

#model = Model('./models/rus_model.udpipe')
#subsent_tokenze(text, model)
