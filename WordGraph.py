import networkx as nx
import matplotlib.pyplot as plt
from conllu import parse
from collections import defaultdict

class link:
    '''    
        It's similar to c-like structure. 
        This class represents a node in sentence dependancy graph
        Init params is got from ConLLU-2014
    '''
    def __init__(self, id, head, word, pos, deprel, form, morpho):
        self.id = id
        self.word = word
        self.head = head
        self.hlink = None
        self.pos = pos
        self.head_link_attr = deprel
        self.children = []
        self.form = form
        self.morpho = morpho

class WordGraph:
    '''
        This class represents a whole sentence dependency tree 
        and provides functionality to make transformations described in Flippova 2008 paper
    '''
    def __init__(self, links):
        self.clause_counter = 1
        self.clause_map = [[]]

        self.links = links
        self._fill_children()
        self._fill_head_links()
        self._add_root()
        self.split()
        self.generate()

    def orig_dtype(self, node):
        dtype= self.links[node].hlink.head_link_attr 
        if dtype==u"conj":  
            return self.orig_dtype(self.links[node].hlink.id)
        else: return dtype, self.links[node].morpho

    def DFS(self, node, idx, indexes, hier):

        types = "ccomp advcl acl:relcl".split()
        dtype = self.links[node].head_link_attr
        new_idx = idx
        if dtype in types: # this is a new clause, use idx + '.' + next number
            if dtype == u"ccomp" and (u"Case=Gen" in self.links[node].morpho or u"Case=Tra" in self.links[node].morpho):
                pass
            suffix = max(hier.get(idx)) if idx in hier else 0
            new_idx = idx + "." + str(suffix + 1)
            hier[idx].add(suffix + 1)
            self.clause_counter += 1
            self.clause_map.append([])
        elif dtype==u"conj": # coordination of two subclauses or coordination from sentence root, split from last '.' and add next number
            t, _ = self.orig_dtype(node) 
            if t in types or t == u"root":
                if dtype == u"ccomp" and (u"Case=Gen" in self.links[node].morpho or u"Case=Tra" in self.links[node].morpho):
                    pass
                id, suffix = idx.rsplit(u".",1)
                suffix = max(hier.get(id)) if id in hier else 1
                new_idx = id + "." + str(suffix + 1)
                hier[id].add(suffix + 1)
                self.clause_counter = 1
                self.clause_map.append([])
        # assign the idx for the node
        indexes[node] = new_idx
        self.clause_map[-1].append(self.links[node].id)
        for child in sorted(self.links[node].children, key=lambda x: x.id): 
            self.DFS( child.id, new_idx, indexes, hier)

    def split(self, ):
        count=1
        # now dictionary is ready, start search from root
        indexes={}
        hier=defaultdict(lambda:set()) # wipe the set
        self.DFS(0,str(count)+'.1',indexes,hier)
               
    def find(self, match, attr):
        '''
        Find nodes depending on the given property and attributes 
        '''
        ans = []
        for i in self.links:
            if match == getattr(i, attr):
                ans.append(i)
        return ans

    def _add_root(self,):
        '''
            Add root node explicity
        '''
        root = link(0, 0, 'root', '', 'root', 'root', '')
        root.hlink = root
        root.children.append(self.find(0, 'head')[0])
        self.links.insert(0, root)      


    def _fill_head_links(self,):
        '''
            Chains the links head by head with its link:
            fill the head field in link instance
        '''
        for i in self.links:
            if i.head != 0:
                i.hlink = self.find(i.head, 'id')[0]
        
    def _fill_children(self):
        '''
            Fill the child field in link instance with all children nodes
        '''
        for i in self.links:
            if i.head == 0:
                continue
            finded_link = self.find(i.head, 'id')[0]
            finded_link.children.append(i)
        

    def generate(self,):
        '''
            Generate a graph as list of nodes
        '''
        sub_sents = []
        self.clause_map[0].remove(0)
        for sub in self.clause_map:
            new_sub = []
            for i in sorted(sub):
                new_sub.append((self.links[i].form))
            sub_sents.append(' '.join(new_sub))
        return sub_sents
        

""" conllu = parse(open('test.conllu').read())
for j in conllu:
    vericlies = []
    for i in j:
        morph = ''
        if i['feats'] != None:
            morph = "|".join([k+'='+v for k,v in i['feats'].items() ])
        vericlies.append( link( i['id'], i['head'], i['lemma'],\
                         i['upostag'], i['deprel'], i['form'],\
                              morph) ) """
    #print() #"|".join([k+'='+v for k,v in i[0]['feats'].items() ])
    #"|".join(m for m in [token[CPOS],token[FEAT]])
#print(WordGraph(vericlies).clause_map)

