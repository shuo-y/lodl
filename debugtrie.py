
class TrieNode:
    def __init__(self):
        self.isend = False
        self.children = [None for _ in range(26)]

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        c = self.root
        for w in word:
            c.children[ord(w) - ord("a")] = TrieNode()
            c = c.children[ord(w) - ord("a")]
        c.isend = True
        
    def shortestrootorfull(self, word):
        c = self.root
        for cnt, w in enumerate(word):
            if c.children[ord(w) - ord("a")] == None:
                return word
            c = c.children[ord(w) - ord("a")]
            if c.isend == True:
                return word[:(cnt + 1)]
        
        return word


