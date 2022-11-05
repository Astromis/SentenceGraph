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


