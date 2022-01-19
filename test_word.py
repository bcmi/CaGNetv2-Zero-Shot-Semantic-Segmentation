# coding:utf-8
import numpy as np
import pickle
import binascii
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin-1')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


data_dict = {'pascal': [
        'background',    # class 0
        'airplane',     # class 1
        'bike',       # class 2
        'bird',          # class 3
        'boat',          # class 4
        'bottle',        # class 5
        'bus',           # class 6
        'car',           # class 7
        'cat',           # class 8
        'chair',         # class 9
        'cow',           # class 10
        'table',   # class 11
        'dog',           # class 12
        'horse',         # class 13
        'motorbike',     # class 14
        'person',        # class 15
        'plant',  # class 16
        'sheep',         # class 17
        'sofa',          # class 18
        'train',         # class 19
        'tv',    # class 20
    ],

'context' : [
        'background',
        'airplane',    # class #0
        'bicycle',      # class #1
        'bird',         # class #2
        'boat',         # class #3
        'bottle',       # class #4
        'bus',          # class #5
        'car',          # class #6
        'cat',          # class #7
        'chair',        # class #8
        'cow',          # class #9
        'table',  # class #10
        'dog',          # class #11
        'horse',        # class #12
        'motorbike',    # class #13
        'person',       # class #14
        'plant',  # class #15
        'sheep',        # class #16
        'sofa',         # class #17
        'train',        # class #18
        'tv',    # class #19
        'sky',          # class #20
        'grass',        # class #21
        'ground',       # class #22
        'road',         # class #23
        'building',     # class #24
        'tree',         # class #25
        'water',        # class #26
        'mountain',     # class #27
        'wall',         # class #28
        'floor',        # class #29
        'track',        # class #30
        'keyboard',     # class #31
        'ceiling',      # class #32
    ],
    'NYU': [
        'Wall',
        'Floor',
        'Cabinet',
        'Bed',
        'Chair',
        'Sofa',
        'Table',
        'Door',
        'window',
        'bookshelf',
        'Picture',
        'Counter',
        'Blinds',
        'Desks',
        'Shelves',
        'Curtain',
        'Dresser',
        'Pillow',
        'Mirror',
        'Floor-mat',
        'Clothes',
        'Ceiling',
        'Books',
        'Refrigerator',
        'Television',
        'Paper',
        'Towel',
        'Shower-curtain',
        'Box',
        'whiteboard',
        'Person',
        'NightStand',
        'Toilet',
        'Sink',
        'Lamp',
        'bathtub',
        'Bag',
        'Other-structure',
        'Other-furniture',
        'Other-prop',
    ],}
##############################################################
root = '/users/zhijiang'
obj = root+'/train/obj_id.csv'
test_obj = root+'/test/test_obj.csv'
train_list = root+'/train/train_obj.csv'
class_list = []
unseen_list = []
unseen_name = []

obj_real_name = {}


class_list = data_dict['pascal']
print (class_list,len(class_list))





##################################################################

def load_txt(class_list):
    '''
    save = []
    
    
    not_match = []
    #class_list = load_labelnames()
    #print (class_list[134])
    #class_list[31] = 'Night-Stand'
    #class_list[168] = 'water drop'
    #final_list = []
    glove_dict = {}
    class_list[2] = 'turtle'
    class_list[220] = 'sealion'
    class_list[9] = 'trafficlight'
    class_list[14] = 'washer'
    class_list[18] = 'hotdog'
    with open('/users/pretrained_word/crawl-300d-2M.vec', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            tmp = line.split(' ')[1:]
            for t in range(len(tmp)):
                tmp[t] = float(tmp[t])
            if line.split(' ')[0] in class_list:
                glove_dict[line.split(' ')[0]] = tmp
    #print (glove_dict['sea'])
    for t in range(len(class_list)):
        cl = class_list[t]
        if cl in glove_dict.keys():
            save.append(glove_dict[cl])
            #final_list.append([cl,cl,0])
        #elif cl.replace("-", " ") in glove_dict.keys():
        #    save.append(glove_dict[cl.replace("-", " ")])
        #    final_list.append(cl.replace("-", " "))
        elif cl.replace(" ", "-") in glove_dict.keys():
            save.append(glove_dict[cl.replace(" ", "-")])
            #final_list.append([cl,cl.replace(" ", "-"),0])
        elif cl.replace(" ", "") in glove_dict.keys():
            save.append(glove_dict[cl.replace(" ", "")])
            #final_list.append([cl,cl.replace(" ", ""),0])
        elif cl.replace("-", "") in glove_dict.keys():
            save.append(glove_dict[cl.replace("-", "")])
            #final_list.append([cl,cl.replace("-", ""),0])
        else:
            if '-' in cl:
                #if cl.split("-")[0] not in glove_dict.keys() or cl.split("-")[1] not in glove_dict.keys():
                #print (cl.split("-")[0],cl.split("-")[1])
                #print (glove_dict[cl.split("-")[0]],glove_dict[cl.split("-")[1]])
                #break
                save.append((np.array(glove_dict[cl.split("-")[0]])+(np.array(glove_dict[cl.split("-")[1]])))/2)
                #print (((np.array(glove_dict[cl.split("-")[0]])+(np.array(glove_dict[cl.split("-")[1]])))/2).shape)
                #final_list.append([cl,cl,1])
            elif ' ' in cl:
                #if cl.split(" ")[0] not in glove_dict.keys() or cl.split(" ")[1] not in glove_dict.keys():
                #print (cl.split(" ")[0],cl.split(" ")[1])

                save.append((np.array(glove_dict[cl.split(" ")[0]])+(np.array(glove_dict[cl.split(" ")[1]])))/2)
                #print (((np.array(glove_dict[cl.split(" ")[0]])+(np.array(glove_dict[cl.split(" ")[1]])))/2).shape)
                #final_list.append([cl,cl,1])
            else:
                not_match.append(t)
                print (t,cl)


    save = np.array(save)
    print (save.shape)
    save_obj(save, 'fasttext')
    #######################################################
    print ('fasttext done!!')
    '''
    '''
    save = []
    
    
    not_match = []
    glove_dict = {}
    with open('/users/pretrained_word/glove.840B.300d.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(' ')
            #print (tmp[0],len(tmp),tmp[-1])
            #break
            glove_dict[tmp[0]] = list(map(float,tmp[1:]))
    
    for t in range(len(class_list)):
        cl = class_list[t]
        if cl in glove_dict.keys():
            save.append(glove_dict[cl])
            #final_list.append([cl,cl,0])
        #elif cl.replace("-", " ") in glove_dict.keys():
        #    save.append(glove_dict[cl.replace("-", " ")])
        #    final_list.append(cl.replace("-", " "))
        elif cl.replace(" ", "-") in glove_dict.keys():
            save.append(glove_dict[cl.replace(" ", "-")])
            #final_list.append([cl,cl.replace(" ", "-"),0])
        elif cl.replace(" ", "") in glove_dict.keys():
            save.append(glove_dict[cl.replace(" ", "")])
            #final_list.append([cl,cl.replace(" ", ""),0])
        elif cl.replace("-", "") in glove_dict.keys():
            save.append(glove_dict[cl.replace("-", "")])
            #final_list.append([cl,cl.replace("-", ""),0])
        else:
            if '-' in cl:
                #if cl.split("-")[0] not in glove_dict.keys() or cl.split("-")[1] not in glove_dict.keys():
                #print (cl.split("-")[0],cl.split("-")[1])
                #print (glove_dict[cl.split("-")[0]],glove_dict[cl.split("-")[1]])
                #break
                save.append((np.array(glove_dict[cl.split("-")[0]])+(np.array(glove_dict[cl.split("-")[1]])))/2)
                #print (((np.array(glove_dict[cl.split("-")[0]])+(np.array(glove_dict[cl.split("-")[1]])))/2).shape)
                #final_list.append([cl,cl,1])
            elif ' ' in cl:
                #if cl.split(" ")[0] not in glove_dict.keys() or cl.split(" ")[1] not in glove_dict.keys():
                #print (cl.split(" ")[0],cl.split(" ")[1])

                save.append((np.array(glove_dict[cl.split(" ")[0]])+(np.array(glove_dict[cl.split(" ")[1]])))/2)
                #print (((np.array(glove_dict[cl.split(" ")[0]])+(np.array(glove_dict[cl.split(" ")[1]])))/2).shape)
                #final_list.append([cl,cl,1])
            else:
                not_match.append(t)
                print (t,cl)
	

    save = np.array(save)
    print (save.shape)
    save_obj(save, 'fasttext')
	#######################################################
    print ('glove done!!')
    '''
    

    save = []
    
    
    not_match = []
    from gensim.models import KeyedVectors
    glove_dict = KeyedVectors.load_word2vec_format('/users/pretrained_word/GoogleNews-vectors-negative300.bin', binary=True, limit=200000)
    #class_list[154] = 'sofa'
    #class_list[156] = 'stabilization wheelbarrow'
    #class_list[216] = 'flower pot'
    #class_list[234] = 'ring cake'
    for t in range(len(class_list)):
        cl = class_list[t]
        if cl in glove_dict.vocab.keys():
            save.append(glove_dict[cl])
            #final_list.append([cl,cl,0])
        #elif cl.replace("-", " ") in glove_dict.keys():
        #    save.append(glove_dict[cl.replace("-", " ")])
        #    final_list.append(cl.replace("-", " "))
        elif cl.replace(" ", "-") in glove_dict.vocab.keys():
            save.append(glove_dict[cl.replace(" ", "-")])
            #final_list.append([cl,cl.replace(" ", "-"),0])
        elif cl.replace(" ", "") in glove_dict.vocab.keys():
            save.append(glove_dict[cl.replace(" ", "")])
            #final_list.append([cl,cl.replace(" ", ""),0])
        elif cl.replace("-", "") in glove_dict.vocab.keys():
            save.append(glove_dict[cl.replace("-", "")])
            #final_list.append([cl,cl.replace("-", ""),0])
        else:
            if '-' in cl:
                #if cl.split("-")[0] not in glove_dict.keys() or cl.split("-")[1] not in glove_dict.keys():
                #print (cl.split("-")[0],cl.split("-")[1])
                #print (glove_dict[cl.split("-")[0]],glove_dict[cl.split("-")[1]])
                #break
                save.append((np.array(glove_dict[cl.split("-")[0]])+(np.array(glove_dict[cl.split("-")[1]])))/2)
                #print (((np.array(glove_dict[cl.split("-")[0]])+(np.array(glove_dict[cl.split("-")[1]])))/2).shape)
                #final_list.append([cl,cl,1])
            elif ' ' in cl:
                #if cl.split(" ")[0] not in glove_dict.keys() or cl.split(" ")[1] not in glove_dict.keys():
                #print (cl.split(" ")[0],cl.split(" ")[1])

                save.append((np.array(glove_dict[cl.split(" ")[0]])+(np.array(glove_dict[cl.split(" ")[1]])))/2)
                #print (((np.array(glove_dict[cl.split(" ")[0]])+(np.array(glove_dict[cl.split(" ")[1]])))/2).shape)
                #final_list.append([cl,cl,1])
            else:
                not_match.append(t)
                print (t,cl)


    save = np.array(save)
    print (save.shape)
    t = save[1:]
    print (save[0][:10],save[1][:10],save[3][:10])
    #save_obj(save, 'word2vector')
    ta = np.linalg.norm(t, ord=2, axis=1, keepdims=True)
    t = t/ta
    print (t[0][:10],t[1][:10],t[3][:10])

    print ('word2vector done!!')
    
load_txt(class_list)
############################################################
a_word = load_obj('./dataset/voc12/word_vectors/word2vec')
print (a_word.shape)
print (a_word[0][:10],a_word[1][:10],a_word[3][:10])
b_word = a_word[1:]
tb = np.linalg.norm(b_word, ord=2, axis=1, keepdims=True)
b_word = b_word/tb
print (b_word[0][:10],b_word[1][:10],b_word[3][:10])
