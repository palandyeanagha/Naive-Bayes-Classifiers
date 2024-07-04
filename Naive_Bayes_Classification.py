# -*- coding: utf-8 -*-

#Importing libraries
import string
from nltk.corpus import stopwords

#defining functions
#load the input file in a desired format - here as a dictionary
def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

#functionto remove stopwords from the dataset
def remove_stop_words(data):
    objects_wo_stopwords = []
    for sentence in data:
        temp_list = []
        for word in sentence.split():
            if word not in stopwords:
                temp_list.append(word)
        objects_wo_stopwords.append(' '.join(temp_list))
    return objects_wo_stopwords

#function to calculate the probability of a word being deceptive
def prob_word_given_decep(word, data, deceptive_word_count):
    if word in deceptive_word_count:
        prob_dece = float(deceptive_word_count[word]/sum_d)
    else:
        prob_dece = 0
    return prob_dece

#function to calculate the probability of a word being truthful
def prob_word_given_true(word, data, truthful_word_count):
    if word in truthful_word_count:
        prob_tru = float(truthful_word_count[word]/sum_t)
    else:
        prob_tru = 0
    return prob_tru




# Load the training data 
train_data = load_file("--Path--/deceptive.train.txt")
train_data.keys()

#collect all words in each line from train_data['objects']

objects = train_data['objects'].copy()

#Remove all punctuations from the data and change case to lower
for i in range(0,len(train_data['objects'])):
    train_data['objects'][i] =  train_data['objects'][i].translate(str.maketrans('','',string.punctuation))
    train_data['objects'][i] = train_data['objects'][i].lower()
    # print(i)
    # print(i)

# List of all stopwords available in the nltk library for English language
stopwords = stopwords.words('english')


#Remove stopwords from your training data
data_wo_stopwords = remove_stop_words(train_data['objects']) 


#Create a vocabulary dictionary to store the words and their counts
vocabulary = dict()
for line in range(0, len(data_wo_stopwords)):
    #line = 0
    #print(line)
    #print(train_data['objects'][line])
    for each in data_wo_stopwords[line].split():
        key = each
        if key in vocabulary:
            vocabulary[key] += 1 
        else:
            vocabulary[key] = 1

#get the word count for all the labelled training data
deceptive_word_count = dict()
truthful_word_count = dict()    

for sentence in range(0, len(data_wo_stopwords)):
    words = data_wo_stopwords[sentence].split()
    
    #for all 'truthful' reviews, add the count in truthful_word_count
    if train_data['labels'][sentence] == 'truthful': 
        for each in words:                
            if each in truthful_word_count:
                truthful_word_count[each] += 1
            else:
                truthful_word_count[each] = 1
                
    #for all 'deceptive' reviews, add the count in deceptive_word_count            
    if train_data['labels'][sentence] == 'deceptive':
        for each in words:
            if each in deceptive_word_count:
                deceptive_word_count[each] += 1
            else:
                deceptive_word_count[each] = 1
                
                
#------------------------------------------------------------------
#work on the test data set
#load test data

test_data = load_file("--Path--/deceptive.test.txt")

#data preprocessing
#Remove all punctuationsconvert test data to lower case
test_lower = test_data['objects'].copy()

for i in range(0,len(test_lower)):
    test_lower[i] =  test_lower[i].translate(str.maketrans('','',string.punctuation))
    test_lower[i] = test_lower[i].lower()

    
#Remove stopwords from your training data
test_wo_stopwords = remove_stop_words(test_lower)

#Using Bayes theorem to solve the problem statement
#Probability of True
n = float(len(data_wo_stopwords))
true_count = float(len([i for i in train_data['labels'] if i=='truthful']))
proba_of_true = float(true_count/n)

#Probability of deceptive
decep_count = float(len([i for i in train_data['labels'] if i=='deceptive']))
proba_of_decep = float(decep_count/n)



#sum of all deceptive words
sum_d = 0
for d, dk in deceptive_word_count.items():
    sum_d = sum_d + deceptive_word_count[d]

#sum of all truthful words        
sum_t = 0
for t, tk in truthful_word_count.items():
    sum_t = sum_t + truthful_word_count[t]

#calculate probability of a word
proba_word = {}
for sent in range(0, len(test_wo_stopwords)):
    words = test_wo_stopwords[sent].split()
    for e in words:
        if e in deceptive_word_count:
            prob1 = float(deceptive_word_count[e]/sum_d)
            #print(prob1)
        else:
            prob1 = 0
        if e in truthful_word_count:
            prob2 = float(truthful_word_count[e]/sum_t)
        else:
            prob2 = 0
        prob_total = float(prob1 + prob2)
        proba_word[e] = prob_total
 

     
#Use the below formula
#p(truthful | word)  = ((p(word | truthful) * p(truthful))/p(word))
#p(deceptive | word)  = ((p(word | deceptive) * p(deceptive))/p(word))

proba_true_given_word = {}
proba_decep_given_word = {}
for sen in range(0, len(test_wo_stopwords)):
    for w in test_wo_stopwords[sen].split():
        try:
            proba_true_given_word[w] = float((prob_word_given_true(w,test_wo_stopwords,truthful_word_count) * proba_of_true)/(prob_word_given_true(w,test_wo_stopwords,truthful_word_count) + prob_word_given_decep(w,test_wo_stopwords,deceptive_word_count)))
        except:
            continue
        
        try:
            proba_decep_given_word[w] = float((prob_word_given_decep(w,test_wo_stopwords,deceptive_word_count) * proba_of_decep)/(prob_word_given_true(w,test_wo_stopwords,truthful_word_count) + prob_word_given_decep(w,test_wo_stopwords,deceptive_word_count)))
            
        except:
            continue
        
#Deciding the propability of the review being truthful or deceptive        
res = {}        
for sen in range(0, len(test_wo_stopwords)):
    all_true_probs = 0
    all_decep_probs = 0
    for w in test_wo_stopwords[sen].split():
        if w in proba_true_given_word:
            all_true_probs += proba_true_given_word[w]
        if w in proba_decep_given_word:
            all_decep_probs += proba_decep_given_word[w]
        
        if all_true_probs > all_decep_probs:
            res[sen] = "Truthful"
        elif all_decep_probs > all_true_probs:
            res[sen] = "Deceptive"
        elif (all_decep_probs == all_true_probs):
            res[sen] = "Can't decide"
            
#Write the results to a file    
text_file = open("--Path--/tested_labels2.txt", "w")
for i in res:
    text_file.write("%s\n" % res[i])
text_file.close()         
        

count = 0
for i in range(0, len(res)):
    if res[i].lower() == test_data['labels'][i].lower():
        count += 1
        
#Accuracy = True Positive / (True Positive+True Negative)*100              
accuracy = count/(len(test_data['labels']))*100.0
                    
                
        