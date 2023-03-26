import pymorphy2

MORPHER = pymorphy2.MorphAnalyzer()


PUNCT_MAP = [('!', '*excl*'), ('"', '*d.qu*'), ('#', '*shar*'), ('$', '*doll*'), 
             ('%', '*perc*'), ('&', '*ampe*'), ("'", '*s.qu*'), ('(', '*l.br*'),
             (')', '*r.br*'), ('*', '*aste*'), ('+', '*plus*'), (',', '*comm*'), 
             ('-', '*minu*'), ('.', '*dot*'), ('/', '*slas*'), (':', '*colo*'),
             (';', '*s.co*'), ('<', '*less*'), ('=', '*equa*'), ('>', '*more*'), 
             ('?', '*ques*'), ('@', '*at*'), ('[', '*l.s.br*'), ('\\', '*b.sl*'), 
             (']', '*r.s.br*'), ('^', '*up*'), ('_', '*u.sc*'), ('`', '**'), 
             ('{', '*l.c.br*'), ('|', '*s.sl*'), ('}', '*r.c.br*'), ('~', '*tild*')]

UD2MYSTEM_GENDER_MAP = {
    "Fem" : "femn",
    "Masc": "masc",
    "Neut": "neut"
}

def punk_repl(str_):

    for punk, val in PUNCT_MAP:
        str_ = str_.replace(punk, val)
    return str_


def get_property(z, j):
    if j == ("form" or "lemma"):
        return {x: punk_repl(y) for x, y in z.data.items() if x == j or len(j) == 0}
    else:
        return {x: y for x, y in z.data.items() if x in j or len(j) == 0}

def inflect_np(base_word: 'WordVertex', grammemes):
    """Inflect noun phrase by given grammemes. See list of grammemes at:
    https://pymorphy2.readthedocs.io/en/stable/user/grammemes.html#grammeme-docs

    Args:
        base_word (WordVertex): The root of the noun phrase
        grammemes (set): A set of grammemes

    Returns:
        str: An inflected string
    """
    processed = ""
    normilized_kw = []        
    base_word_gender = UD2MYSTEM_GENDER_MAP[ base_word.morpho.gender ]
    inflected_word = MORPHER.parse(base_word.form)[0].inflect(grammemes).word
    normilized_kw.append( (base_word.wid, inflected_word))
    for base_child in base_word.children:
        if base_child.pos == "ADJ":
            grammemes.add(base_word_gender)
            inflected_word = MORPHER.parse(base_child.lemma)[0].inflect(frozenset(grammemes)).word
            normilized_kw.append( (base_child.wid, inflected_word))
        else:
            normilized_kw.append( (base_child.wid, base_child.form))
        for other_child in base_child.children:
            normilized_kw.append( (other_child.wid, other_child.form) )

    normilized_kw.sort(key=lambda x: x[0])
    processed = ' '.join([x[1] for x in normilized_kw])
    return processed