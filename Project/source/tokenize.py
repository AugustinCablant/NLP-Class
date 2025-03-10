from collections import Counter
import numpy as np

class Tokenizer:
    def __init__(self, n_steps=1000):
        """
        Initializes the tokenizer with the number of merging steps.
        
        Parameters:
        - n_steps (int): Number of iterations for merging subwords.
        """
        self.vocab = None
        self.steps = None
        self.is_fitted = False
        self.n_steps = n_steps
        self.punc = set([".", " ", "!", ",", "'", ';', '?', '"'])
        
    def split_text(self, text):
        """
        Splits the text into individual characters.
        
        Parameters:
        - text (str): Input text to be tokenized.
        
        Returns:
        - list: List of individual characters from the text.
        """
        return [c for c in text]
        
    def get_best_pair(self, corpus):
        """
        Finds the most frequent adjacent character pair in the corpus.
        
        Parameters:
        - corpus (list): List of characters representing the tokenized text.
        
        Returns:
        - str or None: The most frequent character pair, or None if no valid pair is found.
        """
        if not corpus or len(corpus) < 2:
            return None
        
        subwords = Counter()
        for i in range(len(corpus) - 1):
            if (corpus[i] in self.punc) or (corpus[i+1] in self.punc):
                continue
            subwords[corpus[i] + corpus[i+1]] += 1
        
        return max(subwords, key=subwords.get) if subwords else None
    
    def merge_pairs(self, corpus, best):
        """
        Merges the most frequent character pair into a single token.
        
        Parameters:
        - corpus (list): List of characters representing the tokenized text.
        - best (str): The most frequent character pair to merge.
        
        Returns:
        - list: Updated corpus with merged tokens.
        """
        if not best or best not in ''.join(corpus):
            return corpus
        
        new_corpus = []
        i = 0
        while i < len(corpus) - 1:
            if corpus[i] + corpus[i+1] == best:
                new_corpus.append(best)
                i += 2
            else:
                new_corpus.append(corpus[i])
                i += 1
        
        if i == len(corpus) - 1:
            new_corpus.append(corpus[-1])
        
        return new_corpus
        
    def train(self, text):
        """
        Trains the tokenizer using Byte Pair Encoding.
        
        Parameters:
        - text (str): Input text to train the tokenizer on.
        
        Returns:
        - numpy array: One-hot encoded representation of the final tokenized text.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        corpus = self.split_text(text)
        self.steps = []
        
        for _ in range(self.n_steps):
            best = self.get_best_pair(corpus)
            if not best:
                break
            
            self.steps.append(best)
            corpus = self.merge_pairs(corpus, best)
        
        self.vocab = list(set(corpus))
        self.vocab.append('<UNK>')
        self.is_fitted = True
        
        return self.to_array(corpus)
    
    def encode(self, text):
        """
        Encodes input text using the trained tokenizer.
        
        Parameters:
        - text (str): The text to encode.
        
        Returns:
        - numpy array: One-hot encoded representation of the tokenized text.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call train() first.")
        
        corpus = self.split_text(text)
        for step in self.steps:
            corpus = self.merge_pairs(corpus, step)
        
        corpus = [c if c in self.vocab else '<UNK>' for c in corpus]
        return self.to_array(corpus)

    def to_array(self, corpus):
        """
        Converts the tokenized text into a one-hot encoded numpy array.
        
        Parameters:
        - corpus (list): The tokenized text as a list of tokens.
        
        Returns:
        - numpy array: One-hot encoded representation of the tokenized text.
        """
        # One-hot encode the token indices
        id_list = [self.vocab.index(token) for token in corpus]
        vocab_size = len(self.vocab)
        
        # Create one-hot encoded array
        array = np.zeros((len(id_list), vocab_size), dtype=float)
        for i, id_val in enumerate(id_list):
            array[i, id_val] = 1.0
        
        return array

    def decode(self, array):
        """
        Decodes a one-hot encoded array back into text.
        
        Parameters:
        - array (numpy array): One-hot encoded token representation.
        
        Returns:
        - str: Decoded text.
        """
        # Convert one-hot array back to token indices
        id_list = np.argmax(array, axis=1)

        # Convert token IDs back to tokens
        tokens = [self.vocab[id] for id in id_list]
        
        # Join tokens into a single string
        return ''.join(tokens)