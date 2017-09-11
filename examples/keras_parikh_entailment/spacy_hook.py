from keras.models import model_from_json
import numpy
import numpy.random
import json
from spacy.tokens.span import Span

try:
    import cPickle as pickle
except ImportError:
    import pickle


class KerasSimilarityShim(object):
    @classmethod
    def load(cls, path, nlp, get_features=None, max_length=100):
        # print("get features")
        # print (get_features)
        if get_features is None:
            get_features = get_word_ids
        print("Model path " + str(path))
        with open(path + 'config.json') as file_:
            model = model_from_json(file_.read())
            # print ("HUH")
        with open(path + 'model','rb') as file_:
            weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + weights)
        return cls(model, get_features=get_features, max_length=max_length)

    def __init__(self, model, get_features=None, max_length=100):
        self.model = model
        self.get_features = get_features
        self.max_length = max_length

    def __call__(self, doc):
    # def __call__(self, doc, doc1_style, doc2_style, doc1_TWP, doc2_TWP):
        doc.user_hooks['similarity'] = self.predict
        doc.user_span_hooks['similarity'] = self.predict
        doc.user_token_hooks['similarity'] = self.predict
        return doc


    def predict(self, doc1, array_doc):
        doc2, doc1_style, doc2_style, doc1_TWP, doc2_TWP = array_doc
        # print("type ", type(doc2), type(doc1_style), type(doc2_style), type(doc1_TWP), type(doc2_TWP))
        # print ("doc 1 : " + str(doc1))
        # print(self.get_features)
        x1 = self.get_features([doc1], [doc1_style], [doc1_TWP], max_length=self.max_length, tree_truncate=False)

        x2 = self.get_features([doc2], [doc2_style], [doc2_TWP], max_length=self.max_length, tree_truncate=False)
        scores = self.model.predict([x1, x2])
        return scores[0]


def get_embeddings(vocab, nr_unk=100):

    pos_array = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ','DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON','PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X','CCONJ']
    pos_dict = {value:key for key,value in enumerate(pos_array)}
    
    styling_feature_array = ['is_totally_bold_false','is_totally_bold_true','is_totally_italic_false','is_totally_italic_true','has_real_bullet_symbol_false','has_real_bullet_symbol_true']
    styling_feature_dict = {value:key for key,value in enumerate(styling_feature_array)}
    
    nr_vector = max(lex.rank for lex in vocab) + 1

    vectors = numpy.zeros((nr_vector+nr_unk+2 + len(pos_array) + len(styling_feature_array) + 101, vocab.vectors_length), dtype='float32')

    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank+1] = lex.vector / lex.vector_norm

    for item in pos_dict:
        vectors[nr_vector + nr_unk + 2 + pos_dict[item]][pos_dict[item]] = 1
    for item in styling_feature_dict:
        vectors[nr_vector + nr_unk + 2 + len(pos_array)+ styling_feature_dict[item]][styling_feature_dict[item]+100] = 1
    for i in range(101):
        vectors[nr_vector + nr_unk + 2 + len(pos_array)+len(styling_feature_array)+i][150+i] = 1

    return vectors


def get_word_ids(docs, styling_array, TWP, rnn_encode=False, tree_truncate=False, max_length=100, nr_unk=100):
    pos_array = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ','DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON','PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X','CCONJ']
    pos_dict = {value:key for key,value in enumerate(pos_array)}
    nr_vector = 57392 #1070992 vectors large #57866 alpha1
    styling_feature_array = ['is_totally_bold_false','is_totally_bold_true','is_totally_italic_false','is_totally_italic_true','has_real_bullet_symbol_false','has_real_bullet_symbol_true']
    styling_feature_dict = {value:key for key,value in enumerate(styling_feature_array)}

    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        ar = styling_array[i]
        twp = TWP[i]
        if tree_truncate:
            if isinstance(doc, Span):
                queue = [doc.root]
            else:
                queue = [sent.root for sent in doc.sents]
        else:
            queue = list(doc)
        words = []
        while len(words) <= max_length and queue:
            word = queue.pop(0)
            if rnn_encode or (not word.is_punct and not word.is_space):
                words.append(word)
            if tree_truncate:
                queue.extend(list(word.lefts))
                queue.extend(list(word.rights))
        # words.sort()
        j = 0
        # print 'words: ',words
        for token in words:
            if token.has_vector:
                Xs[i, j] = 0 #token.rank+1
                # print token.orth_,token.rank+1
            else:
                Xs[i, j] = (token.shape % (nr_unk-1))+2
            Xs[i, j+1] = nr_vector + nr_unk + 2 + pos_dict[token.pos_]
            j += 2
            if j >= max_length:
                break
        else:
            Xs[i, 2*len(words)] = 1
            # print ar
            for k,item in enumerate(ar):
                if item:
                    Xs[i,max_length-1-k] = nr_vector + nr_unk + 2 + len(pos_array) + k 
            Xs[i,max_length-1-len(styling_feature_array)] = nr_vector + nr_unk + 2 + len(pos_array)+len(styling_feature_array) + int(twp)
    return Xs


def create_similarity_pipeline(nlp, max_length=210):
    return KerasSimilarityShim.load('/home/ankesh/div_merging_models/alpha1/' + 'similarity/', nlp, max_length=max_length)
    # return [
    #     nlp.tensorizer,
    #     nlp.tagger,
    #     nlp.entity,
    #     nlp.parser,
    #     KerasSimilarityShim.load('/home/ankesh/' + 'similarity/', nlp, max_length=max_length)
    #     # KerasSimilarityShim.load(nlp.path / 'similarity', nlp, max_length=max_length)
    # ]
