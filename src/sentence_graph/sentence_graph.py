from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from conllu import parse
from networkx.drawing.nx_pydot import graphviz_layout
from tqdm.auto import tqdm

from .udpipe_model import Model
from .utils import UD2MYSTEM_GENDER_MAP, get_property


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
    #FIXME: change the name 'head' and hlink to more meaningful one 
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


class Span:
    """The class represents the group of WordVerts. It's not nessesary to be
    a valid UD parsed phrase.

    """
    
    def __init__(self, wv_list: Tuple[List[WordVertex], None], root_idx: Optional[Tuple[int, None]]) -> None:
        self.w_vertices = []
        self.id_set = set()
        if wv_list is not None:
            self.w_vertices = wv_list
            for w in wv_list:
                self.id_set.add(w.wid)
        if root_idx is not None and wv_list is not None:
            self.root = self.w_vertices[root_idx]
        else:
            self.root = None

    def __getitem__(self, key):
        return self.w_vertices[key]

    def __len__(self):
        return len(self.w_vertices) 
    
    def get_source(self):
        """Returns space concatatenated source text 

        Returns:
            str: Soure text
        """
        return " ".join([w.form for w in self.w_vertices])
    
    def find(self, match, attr):
        '''
        Find nodes depending on the given property and attributes 
        '''
        # FIXME: add attr checker
        ans = []
        for i in self.w_vertices:
            if match == i[attr]:
                ans.append(i)
        return ans
    
    def search_by_word(self, word):
        """
        Search the nodes by word 
        (due to some sentence might contain several same words
        this function returns the list)

        """
        return self.find(word, 'form')
    
    
    def remove_node(self, word_id):
        to_remove = self.find(word_id, "wid")[0]
        for i, c in enumerate(to_remove.hlink.children):
            if c.wid == to_remove.wid:
                to_remove.hlink.children.remove(i)
        def remove(wv):
            if not wv.children:
                self.w_vertices.remove(wv.wid)
                return
            for ch in wv.children:
                self.rem(ch)
            self.w_vertices.remove(wv.word_id)
        remove(to_remove)
        #self.w_vertices.remove(word_id)

    def add_node(self, wv: WordVertex):
        if isinstance(wv.head, int):
            AttributeError("To add the node, it has to have a head attribute")
        if wv.hlink < 0 and wv.head > len(self.w_vertices) - 1:
            return ValueError(f"The head attribute must be in range from 0 to max word count ({len(self.w_vertices)} here)")
        wv.wid = len(self.id_set)
        self._add_node(wv)
        found_link = self.w_vertices(wv['head'], 'wid')[0]
        found_link['children'].append(self.w_vertices[-1])
        wv.hlink = found_link
        
    def get_nx(self, property="form"):
        '''
        Get sentence graph as NetworkX DiGraph
        proprerty see plot()
        '''
        g = nx.DiGraph()
        nodes = [(vex['wid'], get_property(vex, property))
                         for vex in self.w_vertices]
        edges = [(x['hlink']['wid'], x['wid'], )
                         for x in self.w_vertices if x['hlink'] != None]
        # don't forget to add a virtual root node
        nodes.append((0, {property : "root"}))
        edges.append((0, self.root.wid))
        nodes.sort(key=lambda x: x[0])
        edges.sort(key=lambda x: x[0])

        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        
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


class SentenceGraph(Span):
    '''
        This class represents a whole sentence dependency tree.
        It takes the sentence writen in conluu-2014 format and produces
        hendfull structure to operate it as a graph.
    '''

    def __repr__(self):
        sent_len = len(self.w_vertices) 
        representation_len = sent_len if sent_len < 5 else 5
        repr_str = "{} " * representation_len
        repr_str += "...<SentenceGraph>"
        return repr_str.format(*self.w_vertices[:sent_len])

    

    def _get_np(self, node, np):
        if node["pos"] == "NOUN":
            np.append(node)
        if len(node["children"]) == 0:
            return
        else:
            for c in node["children"]:    
                self._get_np(c, np)
    
    def get_np(self, start_node=None):
        """Return a list of Noun Phrases of the text

        Args:
            start_node (WordVertex): A start word node
        
        Returns:
            list: Existed noun phrases
        """
        noun_phrases = []
        if start_node != None:
            self._get_np(start_node, noun_phrases)
        else:
            self._get_np(self.root, noun_phrases)
        return noun_phrases


    def _orig_dtype(self, node):
        dtype = self.w_vertices[node]['hlink']['head_link_attr']
        if dtype == u"conj":
            return self._orig_dtype(self.w_vertices[node]['hlink']['wid'])
        else:
            return dtype, self.w_vertices[node]["morpho"]

    def _DFS(self, node, idx, indexes, hier, clause_map):

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
        clause_map[-1].append(self.w_vertices[node]['wid'])
        for child in sorted(self.w_vertices[node]['children'], key=lambda x: x['wid']):
            self._DFS(child['wid'], new_idx, indexes, hier, clause_map)

    def _split_by_clause(self, clause_map):
        '''
        Split the sentence into clauses
        '''
        count = 1
        # now dictionary is ready, start search from root
        indexes = {}
        hier = defaultdict(lambda: set())  # wipe the set
        self._DFS(0, str(count)+'.1', indexes, hier, clause_map)

    def get_clauses(self,):
        '''
            Return a list of clauses
        '''
        clause_map = [[]]
        self._split_by_clause(clause_map)
        sub_sents = []
        
        clause_map[0].remove(0)
        for sub in clause_map:
            new_sub = []
            for i in sorted(sub):
                new_sub.append((self.w_vertices[i]['form']))
            sub_sents.append(' '.join(new_sub))
        return sub_sents
    

        
    def _add_node(self, wv):
        self.w_vertices.append(wv)
        self.id_set.add(wv.wid)



class TextParser:
    '''
    Class-interface that performs the parsing
    text (as string) into list of SentenceGraphs
    '''

    def __init__(self, udpipe_model_path):
        self.model = Model(udpipe_model_path)

    def parse(self, text, syntax_parse=True, verbose=False, mystem_gender_translate=False):
        '''
        Parse the text into list of SentenceGraphs
        '''
        sentence_graphs = []
        sentences = self.model.tokenize(text)
        if verbose:
            sentences = tqdm(sentences, desc="UDPipe parsing")
        for s in sentences:
            self.model.tag(s)
            if syntax_parse:
                self.model.parse(s)
        sents_conllu = self.model.write(sentences, "conllu")
        parsed_conllu = parse(sents_conllu)
        if verbose:
            parsed_conllu = tqdm(parsed_conllu, desc="Building Sentence Graphs")
        for sent in parsed_conllu:
            sentence_graphs.append(self._build_graph(sent, syntax_parse, mystem_gender_translate))

        return sentence_graphs

    def _build_graph(self, conllu_sentence, syntax_parse=True, mystem_translate=False):
        '''
        Build graph from sentence representing in CoNLLU-2014 format
        '''
        verteces = []
        root_idx = None
        for i, w in enumerate(conllu_sentence):
            morph = None
            if w['feats'] != None:
                morph = Morpho(w['feats'])
                if mystem_translate:
                    morph.gender = UD2MYSTEM_GENDER_MAP[morph.gender]
            verteces.append(WordVertex(w['id'], w['head'], w['lemma'], w['upostag'], w['deprel'], w['form'], morph))
            if w["head"] == 0:
                root_idx = i
        sg = SentenceGraph(wv_list=verteces, root_idx=root_idx)

        if syntax_parse:
            #self._add_root(sg)
            self._fill_children_verticles(sg)
            self._fill_head_verticles(sg)
        
        return sg

    def _add_root(self, sg: SentenceGraph):
        '''
            Add root node explicity
        '''
        actual_root_word = sg.find(0, 'head')[0]
        sg.root = actual_root_word

    def _fill_head_verticles(self, sg: SentenceGraph):
        '''
            Chains the WordVertexs head by head with:
            fill the head field in WordVertexs instance
        '''
        for i in sg.w_vertices:
            if i.head != 0:
                i['hlink'] = sg.find(i['head'], 'wid')[0]

    def _fill_children_verticles(self, sg: SentenceGraph):
        '''
            Fill the child field in WordVertexs instance with all children nodes
        '''
        for i in sg.w_vertices:
            if i['head'] == 0:
                continue
            finded_link = sg.find(i['head'], 'wid')[0]
            finded_link['children'].append(i)
