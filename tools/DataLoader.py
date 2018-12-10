import numpy as np


def Loadembedding(filepath):
    with open(filepath,"r",encoding = 'utf-8') as f:
        word_bag_size,embedding_dimention =map(int, f.readline().split())
        embedding = []
        embedding.append([0.]*embedding_dimention)
        word_map = dict()
        cnt = 0
        print("start load embedding")
        for line in f.readlines():
            cnt+=1
            temp_line = line.split()
            word_map[temp_line[0]] = cnt
            embedding.append([float(i) for i in temp_line[1:]])
        embedding = np.array(embedding,dtype =np.float32)
        print("Loaded embeding about {} words and {} dimention!".format(word_bag_size,embedding_dimention))
    return word_map,embedding,embedding_dimention

def LoadText(filepath,word_map):
    data = []
    lable = []
    max_sen_len = -1
    with open(filepath,"r",encoding = 'utf-8') as f:
        for line in f.readlines():
            temp_line = line.split('\t\t')
            sentence = temp_line[-1]
            temp_label = int(temp_line[0])
            temp_sentence = []
            max_sen_len = max(max_sen_len,len(sentence.split()))
            for word in sentence.split():
                if word in word_map:
                    temp_sentence.append(word_map[word])
                else:
                    temp_sentence.append(0)
            data.append(temp_sentence)
            lable.append(temp_label)
    return (data),(lable),max_sen_len


def PADDING(inputs,length):
    flag = 56
    for i in range(len(inputs)):
        if len(inputs[i])<length:
            inputs[i]=inputs[i]+[0]*(length-len(inputs[i]))
        assert len(inputs[i])==flag
    print("padding 完成")
    print(np.shape(np.array(inputs)))
    return np.array(inputs)
