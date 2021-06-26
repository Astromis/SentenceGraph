from conllu import parse
from collections import defaultdict
from udpipe_model import Model
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

class WordVertex:
    '''     
        This class represents a node word in sentence dependancy graph.
        Dictionary data are got from ConLLU-2014
    '''
    __slots__ = ["data"]
    def __init__(self, id: int, head:int, lemma: str, pos:str, deprel:str, form:str, morpho:str):
        self.data = {
        "id" : id,
        "lemma" : lemma,
        "head" : head,
        "hlink" : None,
        "pos" : pos,
        "head_link_attr" : deprel,
        "children" : [],
        "form" : form,
        "morpho" :   morpho,
        }
        
    def __getitem__(self, key):
        return self.data[key]
        
    def __setitem__(self, key, val):
        self.data[key] = val
        
    def __str__(self):
        return self.data['form']

    def __repr__(self):
        return '{}_ID-{}<WordVertex>'.format(self.data['form'], self.data['id'])
    

class SentenceGraph:
    '''
        This class represents a whole sentence dependency tree.
        It takes the sentence writen in conluu-2014 format and produces
        hendfull structure to operate it as a graph.
    '''
    def __init__(self, conllu_sentence):
        self.clause_map = [[]]
        self.links = []

        self._build_graph(conllu_sentence)
        self._add_root()
        self._fill_children_verticles()
        self._fill_head_verticles()
        self._split_by_clause()
    
    def __getitem__(self, key):
        return self.links[key]
    
    def __repr__(self):
        sent_len = len(self.links)
        representation_len =  sent_len if sent_len < 5 else 5
        repr_str = "{} " * representation_len
        repr_str += "...<SentenceGraph>"
        return repr_str.format(*self.links[:sent_len])

    def by_word(self, word):
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
            morph = ''
            if i['feats'] != None:
                morph = "|".join([k+'='+ v for k,v in i['feats'].items() ])
            self.links.append( WordVertex( i['id'], i['head'], i['lemma'], i['upostag'], i['deprel'], i['form'], morph) )
        
    def _orig_dtype(self, node):
        dtype= self.links[node]['hlink']['head_link_attr'] 
        if dtype==u"conj":  
            return self._orig_dtype(self.links[node]['hlink']['id'])
        else: return dtype, self.links[node]["morpho"]

    def _DFS(self, node, idx, indexes, hier):

        types = "ccomp advcl acl:relcl".split()
        dtype = self.links[node]['head_link_attr']
        new_idx = idx
        if dtype in types: # this is a new clause, use idx + '.' + next number
            if dtype == u"ccomp" and (u"Case=Gen" in self.links[node]['morpho'] or u"Case=Tra" in self.links[node]['morpho']):
                pass
            suffix = max(hier.get(idx)) if idx in hier else 0
            new_idx = idx + "." + str(suffix + 1)
            hier[idx].add(suffix + 1)
            self.clause_map.append([])
        elif dtype==u"conj": # coordination of two subclauses or coordination from sentence root, split from last '.' and add next number
            t, _ = self._orig_dtype(node) 
            if t in types or t == u"root":
                if dtype == u"ccomp" and (u"Case=Gen" in self.links[node]['morpho'] or u"Case=Tra" in self.links[node]['morpho']):
                    pass
                id, suffix = idx.rsplit(u".",1)
                suffix = max(hier.get(id)) if id in hier else 1
                new_idx = id + "." + str(suffix + 1)
                hier[id].add(suffix + 1)
                self.clause_map.append([])
        # assign the idx for the node
        indexes[node] = new_idx
        self.clause_map[-1].append(self.links[node]['id'])
        for child in sorted(self.links[node]['children'], key=lambda x: x['id']): 
            self._DFS( child['id'], new_idx, indexes, hier)

    def _split_by_clause(self, ):
        '''
        Split the sentence into clauses
        '''
        count=1
        # now dictionary is ready, start search from root
        indexes={}
        hier=defaultdict(lambda:set()) # wipe the set
        self._DFS(0,str(count)+'.1',indexes,hier)
               
    def find(self, match, attr):
        '''
        Find nodes depending on the given property and attributes 
        '''
        ans = []
        for i in self.links:
            if match == i[attr]:
                ans.append(i)
        return ans

    def _add_root(self,):
        '''
            Add root node explicity
        '''
        root = WordVertex(0, 0, 'root', '', 'root', 'root', '')
        root['hlink'] = root
        root['children'].append(self.find(0, 'head')[0])
        self.links.insert(0, root)      

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
        for i in self.links:
            #if i.head != 0:
            i['hlink'] = self.find(i['head'], 'id')[0]
        
    def _fill_children_verticles(self):
        '''
            Fill the child field in WordVertexs instance with all children nodes
        '''
        for i in self.links:
            if i['head'] == 0:
                continue
            finded_link = self.find(i['head'], 'id')[0]
            finded_link['children'].append(i)

    def get_clauses(self,):
        '''
            Return a list of clauses
        '''
        sub_sents = []
        self.clause_map[0].remove(0)
        for sub in self.clause_map:
            new_sub = []
            for i in sorted(sub):
                new_sub.append((self.links[i].form))
            sub_sents.append(' '.join(new_sub))
        return sub_sents

    def get_nx(self, proprerties=[]):
        '''
        Plot graph with NetworkX
        proprerty None or list of properties
        '''
        g = nx.DiGraph()
        def property_filter(z, j): 
            return {x:y for x,y in z.data.items() if x in j or len(j) == 0}
        #g.add_nodes_from([(x['id'], {proprerty: x['proprerty']} ) for x in self.links])
        g.add_nodes_from([(vex['id'], property_filter(vex, proprerties) ) for vex in self.links])
        g.add_edges_from([(x['hlink']['id'], x['id'], ) for x in self.links])
        return g
    
    def plot(self, fsize=(16,10), propertie="pos"):
        """
        Currently works with pos, id, morpho
        """
        G = self.get_nx([propertie])
        plt.figure(figsize=fsize)
        H = nx.convert_node_labels_to_integers(G,)
        mapping = {x:G.nodes()[y][propertie] for x,y in enumerate(G.nodes())}
        pos = graphviz_layout(H, prog="dot")
        nx.draw_networkx(G, pos, nodelist=None, labels=mapping)#pos
        plt.show()
        #plt.figure(figsize=fsize)
        #H = nx.convert_node_labels_to_integers(g,)
        #mapping = {x:y for x,y in enumerate(g.nodes())}
        #pos = graphviz_layout(H, prog="dot")
        #nx.draw_networkx(g, pos, nodelist=None)#pos
        #plt.show()

class TextParser:
    '''
    Class-interface that performs the parsing
    text (as string) into list of SentenceGraphs
    '''
    def __init__(self, udpipe_model_path):
        self.model = Model(udpipe_model_path)

    def parse(self, text):
        '''
        Parse the text into list of SentenceGraphs
        '''
        sentence_graphs = []
        sentences = self.model.tokenize(text)
        for s in sentences:
                self.model.tag(s)
                self.model.parse(s)
        sents_conllu = self.model.write(sentences, "conllu")
        for sent in parse(sents_conllu):
            sentence_graphs.append(SentenceGraph(sent))
    
        return sentence_graphs
