#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import numpy as np
import struct
import time

#将标签转换为onehot表示
def change_y_to_onehot(y, n_class = 5):
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[label-1] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

#读入文档
def load_inputs_document(input_file, word_id_file, max_sen_len, max_doc_len, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file

    x, y, sen_len, doc_len = [], [], [], []
    print('loading input {}...'.format(input_file))
    for line in open(input_file):
        line = line.lower().decode('utf8', 'ignore').split('\t\t')
        y.append(int(line[0]))

        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        doc = ' '.join(line[1:])
        sentences = doc.split('<sssss>')
        i = 0
        for sentence in sentences:
            j = 0
            for word in sentence.split():
                if j < max_sen_len:
                    if word in word_to_id:
                        t_x[i, j] = word_to_id[word]
                        j += 1
                else:
                    break
            t_sen_len[i] = j
            i += 1
            if i >= max_doc_len:
                break

        doc_len.append(i)
        sen_len.append(t_sen_len)
        x.append(t_x)

    y = change_y_to_onehot(y)
    print('done!')

    return np.asarray(x), np.asarray(y), np.asarray(sen_len), np.asarray(doc_len)
    
def batch_index(length, batch_size, n_iter=100, test=False):
    index = range(length)
    for j in xrange(n_iter):
        if not test: np.random.shuffle(index)
        for i in xrange(int(length / batch_size)+1):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > length: 
                end = length
                if not test: break
            if start >=length: break
            yield index[start : end]

def load_w2v(w2v_file, embedding_dim, debug=False):
    fp = open(w2v_file)
    words, embedding_dim = map(int, fp.readline().split())
    
    w2v = []
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    word_dict = dict()
    print 'loading word_embedding {}...'.format(w2v_file)
    print 'wordlist: {} embedding_dim: {}'.format(words, embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        word_dict[line[0]] = cnt
        w2v.append([float(v) for v in line[1:]])
    print 'done!\n'
    w2v = np.asarray(w2v, dtype=np.float32)
    if debug:
        print 'shape of w2v:',np.shape(w2v)
        word='the'
        print 'id of \''+word+'\':',word_dict[word]
        print 'vector of \''+word+'\':',w2v[word_dict[word]]
    return word_dict, w2v


def load_data_for_DSCNN_sen(input_file, word_to_id, max_doc_len, n_class=2, n_fold=0, index=1):
    x1, y1, doc_len1 = [], [], []
    x2, y2, doc_len2 = [], [], []

    print 'loading input {}...'.format(input_file)
    for line in open(input_file):
        line = line.split('\t\t')
        wordlist = line[-1].split()
        
        tmp_x = np.zeros((max_doc_len), dtype=np.int)
        i = 0
        for word in wordlist:
            if i >= max_doc_len:
                break
            if word in word_to_id:
                tmp_x[i] = word_to_id[word]
                i += 1
        
        tmp_y = np.zeros((n_class), dtype=np.int)
        tmp_y[int(line[0])-1]=1

        if n_fold and index % n_fold==0:
            x, y, doc_len = x2, y2, doc_len2
        else :
            x, y, doc_len = x1, y1, doc_len1
        x.append(tmp_x)
        y.append(tmp_y)
        doc_len.append(i)
        index+=1

    print 'done!'
    if n_fold:
        return np.asarray(x1), np.asarray(y1), np.asarray(doc_len1), np.asarray(x2), np.asarray(y2), np.asarray(doc_len2)
    else :
        return np.asarray(x1), np.asarray(y1), np.asarray(doc_len1)

    
def load_data_for_DSCNN_doc(input_file, word_to_id, max_doc_len, max_sen_len, n_class=2, n_fold=0, index=1):
    x1, y1, doc_len1, sen_len1 = [], [], [], []
    x2, y2, doc_len2, sen_len2 = [], [], [], []

    print 'loading input {}...'.format(input_file)
    for line in open(input_file):
        line = line.split('\t\t')
        sentences = line[-1].split('<sssss>')
        
        tmp_x = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        tmp_sen = np.zeros((max_doc_len), dtype=np.int)
        i = 0
        for sen in sentences:
            if i >= max_doc_len:
                    break
            words=sen.split()
            j=0
            for word in words:
                if j >= max_sen_len:
                    break
                if word in word_to_id:
                    tmp_x[i,j] = word_to_id[word]
                    j += 1
            if j:
                tmp_sen[i]=j
                i+=1

        tmp_y = np.zeros((n_class), dtype=np.int)
        tmp_y[int(line[0])-1]=1

        if n_fold and index % n_fold==0:
            x, y, doc_len, sen_len = x2, y2, doc_len2, sen_len2
        else :
            x, y, doc_len, sen_len = x1, y1, doc_len1, sen_len1
        x.append(tmp_x)
        y.append(tmp_y)
        doc_len.append(i)
        sen_len.append(tmp_sen)
        index+=1

    print 'done!'
    if n_fold:
        return np.asarray(x1), np.asarray(y1), np.asarray(doc_len1), np.asarray(sen_len1), np.asarray(x2), np.asarray(y2), np.asarray(doc_len2), np.asarray(sen_len2)
    else :
        return np.asarray(x1), np.asarray(y1), np.asarray(doc_len1), np.asarray(sen_len1)

# if __name__ == '__main__':
#     '''
#     word_id_mapping, w2v = load_w2v('../data/Yelp/yelp-2013-vectors.txt',200)
#     x1, y1, doc_len1, x2, y2, doc_len2= load_data_for_DSCNN_sen('doc1.txt', word_id_mapping, max_doc_len=10, n_class=2, n_fold=2, index=2)
#     print 'x1:\n',x1
#     print 'doc_len1:\n',doc_len1
#     print 'y1:\n',y1
#     print 'x2:\n',x2
#     print 'doc_len2:\n',doc_len2
#     print 'y2:\n',y2
#     '''
#     word_id_mapping, w2v = load_w2v('../data/Yelp/yelp-2013-vectors.txt',200)
#     x1, y1, doc_len1, sen_len1 =load_data_for_DSCNN_doc('doc1.txt', word_id_mapping, max_doc_len=2, max_sen_len=3, n_class=2, n_fold=0, index=1)
#     print 'x1:\n',x1
#     print 'doc_len1:\n',doc_len1
#     print 'sen_len1:\n',sen_len1
#     print 'y1:\n',y1
   
