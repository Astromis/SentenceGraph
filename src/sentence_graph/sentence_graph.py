from dataclasses import dataclass
from conllu import parse
from collections import defaultdict
from .udpipe_model import Model
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from networkx.drawing.nx_pydot import graphviz_layout


class Morpho:
    """
    Represents a morphological featues with universal dependencies format
    """

    def __init__(self, feature_dict):
        for k, v in feature_dict.items():
            setattr(self, k.lower(), v)

    """ # nominal
    gender: str = None
    animacy: str = None
    nounclass: str = None
    number: str = None
    case: str = None
    definite: str = None
    degree: str = None
    # verbal
    verbform: str = None
    mood: str = None
    tense: str = None
    aspect: str = None
    voice: str = None
    evident: str = None
    polarity: str = None
    person: str = None
    polite: str = None
    clusivity: str = None
    # lexical features
    prontype: str = None
    numtype: str = None
    poss: str = None
    reflex: str = None
    foreign: str = None
    abbr: str = None
    typo: str = None """


class WordVertex:
    '''     
        This class represents a node word in sentence dependancy graph.
        Dictionary data are got from ConLLU-2014
    '''
    __slots__ = ["data", "wid", "lemma", "head", "hlink",
                 "pos", "head_link_attr", "children", "form", "morpho"]

    def __init__(self, wid: int, head: int, lemma: str, pos: str, deprel: str, form: str, morpho: str):
        # FIXME: It seems that hlink is redundunt
        self.data = {
            "wid": wid,
            "lemma": lemma,
            "head": head,
            "hlink": None,
            "pos": pos,
            "head_link_attr": deprel,
            "children": [],
            "form": form,
            "morpho":   morpho,
        }

        for k, v in self.data.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __str__(self):
        return self.data['form']

    def __repr__(self):
        return '{}_WID-{}<WordVertex>'.format(self.data['form'], self.data['wid'])


class SentenceGraph:
    '''
        This class represents a whole sentence dependency tree.
        It takes the sentence writen in conluu-2014 format and produces
        hendfull structure to operate it as a graph.
    '''

    def __init__(self, conllu_sentence, syntax_parse=True):
        # FIXME: the clause map is not suppose to be here
        self.clause_map = [[]]
        self.w_vertices = []
        self.root = None

        self._build_graph(conllu_sentence)
        if syntax_parse:
            self._add_root()
            self._fill_children_verticles()
            self._fill_head_verticles()
        

    def __getitem__(self, key):
        return self.w_vertices[key+1]

    def __len__(self):
        return len(self.w_vertices) - 1

    def __repr__(self):
        sent_len = len(self.w_vertices) - 1
        representation_len = sent_len if sent_len < 5 else 5
        repr_str = "{} " * representation_len
        repr_str += "...<SentenceGraph>"
        return repr_str.format(*self.w_vertices[1:sent_len])

    def get_source(self):
        """Returns space concatatenated source text 

        Returns:
            str: Soure text
        """
        return " ".join([w.form for w in self.w_vertices[1:]])

    def search_by_word(self, word):
        """
        Search the nodes by word 
        (due to some sentence might contain several same words
        this function returns the list)

        """
        return self.find(word, 'form')

    def _build_graph(self, conllu_sentence):
        '''
        Build graph from sentence representing in CoNLLU-2014 format
        '''
        for i in conllu_sentence:
            morph = None
            if i['feats'] != None:
                # morph = "|".join([k+'='+ v for k,v in i['feats'].items() ])
                morph = Morpho(i['feats'])
            self.w_vertices.append(WordVertex(
                i['id'], i['head'], i['lemma'], i['upostag'], i['deprel'], i['form'], morph))

    def _orig_dtype(self, node):
        dtype = self.w_vertices[node]['hlink']['head_link_attr']
        if dtype == u"conj":
            return self._orig_dtype(self.w_vertices[node]['hlink']['wid'])
        else:
            return dtype, self.w_vertices[node]["morpho"]

    def _DFS(self, node, idx, indexes, hier):

        types = "ccomp advcl acl:relcl".split()
        dtype = self.w_vertices[node]['head_link_attr']
        new_idx = idx
        if dtype in types:  # this is a new clause, use idx + '.' + next number
            if dtype == u"ccomp" and (u"Case=Gen" in self.w_vertices[node]['morpho'] or u"Case=Tra" in self.w_vertices[node]['morpho']):
                pass
            suffix = max(hier.get(idx)) if idx in hier else 0
            new_idx = idx + "." + str(suffix + 1)
            hier[idx].add(suffix + 1)
            self.clause_map.append([])
        elif dtype == u"conj":  # coordination of two subclauses or coordination from sentence root, split from last '.' and add next number
            t, _ = self._orig_dtype(node)
            if t in types or t == u"root":
                if dtype == u"ccomp" and (u"Case=Gen" in self.w_vertices[node]['morpho'] or u"Case=Tra" in self.w_vertices[node]['morpho']):
                    pass
                id, suffix = idx.rsplit(u".", 1)
                suffix = max(hier.get(id)) if id in hier else 1
                new_idx = id + "." + str(suffix + 1)
                hier[id].add(suffix + 1)
                self.clause_map.append([])
        # assign the idx for the node
        indexes[node] = new_idx
        self.clause_map[-1].append(self.w_vertices[node]['wid'])
        for child in sorted(self.w_vertices[node]['children'], key=lambda x: x['wid']):
            self._DFS(child['wid'], new_idx, indexes, hier)

    def _split_by_clause(self, ):
        '''
        Split the sentence into clauses
        '''
        count = 1
        # now dictionary is ready, start search from root
        indexes = {}
        hier = defaultdict(lambda: set())  # wipe the set
        self._DFS(0, str(count)+'.1', indexes, hier)

    def find(self, match, attr):
        '''
        Find nodes depending on the given property and attributes 
        '''
        ans = []
        for i in self.w_vertices:
            if match == i[attr]:
                ans.append(i)
        return ans

    def _add_root(self,):
        '''
            Add root node explicity
        '''
        root = WordVertex(0, 0, 'root', '', 'root', 'root', None)
        root['hlink'] = root
        actual_root_word = self.find(0, 'head')[0]
        self.root = actual_root_word
        root['children'].append(actual_root_word)
        self.w_vertices.insert(0, root)

    """ def remove(self, link_id):
        '''
            Remove node from the graph
        '''
        self.links.remove(link_id) """

    def _fill_head_verticles(self,):
        '''
            Chains the WordVertexs head by head with:
            fill the head field in WordVertexs instance
        '''
        for i in self.w_vertices:
            # if i.head != 0:
            i['hlink'] = self.find(i['head'], 'wid')[0]

    def _fill_children_verticles(self):
        '''
            Fill the child field in WordVertexs instance with all children nodes
        '''
        for i in self.w_vertices:
            if i['head'] == 0:
                continue
            finded_link = self.find(i['head'], 'wid')[0]
            finded_link['children'].append(i)

    def get_clauses(self,):
        '''
            Return a list of clauses
        '''
        self._split_by_clause()
        sub_sents = []
        self.clause_map[0].remove(0)
        for sub in self.clause_map:
            new_sub = []
            for i in sorted(sub):
                new_sub.append((self.w_vertices[i]['form']))
            sub_sents.append(' '.join(new_sub))
        return sub_sents

    def get_nx(self, property="form"):
        '''
        Get sentence graph as NetworkX DiGraph
        proprerty see plot()
        '''
        g = nx.DiGraph()

        # FIXME: functions inside funcion is evel
        def punk_repl(str_):
            punct_map = [
                (',', "*зпт*"),
                ('.', "*тчк*")
            ]
            for punk, val in punct_map:
                str_ = str_.replace(punk, val)
            return str_

        def get_property(z, j):
            if j == ("form" or "lemma"):
                return {x: punk_repl(y) for x, y in z.data.items() if x == j or len(j) == 0}
            else:
                return {x: y for x, y in z.data.items() if x in j or len(j) == 0}
        g.add_nodes_from([(vex['wid'], get_property(vex, property))
                         for vex in self.w_vertices])
        g.add_edges_from([(x['hlink']['wid'], x['wid'], )
                         for x in self.w_vertices])
        return g

    def plot(self, fsize=(16, 10), property="form"):
        """
        Plot graph with NetworkX
        proprerty One of ["form", "pos", "wid", "morpho"] that will be node annotation
        """
        if property not in ["form", "pos", "wid", "morpho"]:
            raise ValueError("Wrong property name")
        G = self.get_nx(property)
        plt.figure(figsize=fsize)
        H = nx.convert_node_labels_to_integers(G,)
        mapping = {x: G.nodes()[y][property] for x, y in enumerate(G.nodes())}
        try:
            pos = graphviz_layout(H, prog="dot")
            nx.draw_networkx(G, pos, nodelist=None, labels=mapping)
        except FileNotFoundError:
            raise ImportError(
                "Looks like you forget to install graphviz program. Please, install it typing in console 'sudo apt install graphviz'")
        plt.show()


class TextParser:
    '''
    Class-interface that performs the parsing
    text (as string) into list of SentenceGraphs
    '''

    def __init__(self, udpipe_model_path):
        self.model = Model(udpipe_model_path)

    def parse(self, text, syntax_parse=True):
        '''
        Parse the text into list of SentenceGraphs
        '''
        sentence_graphs = []
        sentences = self.model.tokenize(text)
        for s in tqdm(sentences, desc="UDPipe parsing"):
            self.model.tag(s)
            if syntax_parse:
                self.model.parse(s)
        sents_conllu = self.model.write(sentences, "conllu")
        for sent in tqdm(parse(sents_conllu), desc="Conllu parsing"):
            sentence_graphs.append(SentenceGraph(sent, syntax_parse))

        return sentence_graphs
