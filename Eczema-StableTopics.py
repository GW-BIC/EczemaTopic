# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:46:50 2014

@author: yijun
"""

import re
import pandas as pd
import ftfy
from nltk.corpus                     import stopwords

from subprocess import Popen, PIPE, STDOUT
from math import log, sqrt
import time
import numpy as np
import os

def taskHas(task):
    
    tasks = [-2]
    #tasks = [12]
    
    if task in tasks:
        print
        print ('Task:', task)
        return True
    return False

mainDir = './data'
postfile= mainDir + '/eczema-post-relavant.csv'
javapath='/usr/bin/'
classpath = './mallet'

if taskHas(-2):
    ## tokenize the post
    word2dftf = {}
    nTokens = 0
    nDocs = 0
    stopwords = stopwords.words('english')
    with open(mainDir + '/en.txt', 'r') as f:
        for line in f:
            stopwords.append(line.strip())
    with open(mainDir + '/extra-stop.txt') as f:
        for line in f:
            stopwords.append(line.strip())

    posts = pd.read_csv(postfile, encoding="iso-8859-1")

    for i in posts.index:
        nDocs += 1
        text = str(posts.iloc[i]['text'])
        newtext = re.sub(r'(\s[^a-z]+\s)|(\s[0-9]+\s)', ' ', (re.sub(r'([.,;?:€/()!~&"])', ' ', text)))
        thisDoc_word2tf = {}
        for w in newtext.split():
            nTokens += 1
            thisDoc_word2tf[w] = thisDoc_word2tf.get(w, 0) + 1
        for w in thisDoc_word2tf:
            if w not in word2dftf:
                word2dftf[w] = [0, 0]
            word2dftf[w][0] += 1
            word2dftf[w][1] += thisDoc_word2tf[w]

    lowdf = list([w for w in word2dftf if word2dftf[w][0] < 10])
    highdf = list([w for w in word2dftf if word2dftf[w][0] > 10])
    print(lowdf)
    unselected = stopwords + lowdf

    with open(mainDir + '/post-cleaned.txt', 'w') as g:
        for i in posts.index:
            text = str(posts.iloc[i]['text'])
            docID = posts.iloc[i]['docID']
            newtext = re.sub(r'(\s[^a-z]+\s)|(\s[0-9]+\s)', ' ', (re.sub(r'([.,;?:€/()!~&"])', ' ', text)))
            words = [w for w in newtext.split() if w not in unselected]
            if not words:
                print('Doc', docID, 'is empty')
            g.write('%s\t%s\n' % (docID, ' '.join(words)))

if taskHas(-1):
    cmd = [ javapath+'java','-Xmx4G','-ea', '-Dfile.encoding=UTF-8', '-classpath',
            classpath+'/bin:'+ classpath +'/lib/mallet-deps.jar',
            'cc.mallet.classify.tui.Csv2Vectors2',
            '--input', mainDir+'/post-cleaned.txt',
            '--output', mainDir+'/post.mallet',
            '--line-regex', r'^(\S*)\t([^\t]*)$',
            '--token-regex','\S+',
            '--label', '0', '--name', '1', '--data', '2', '--keep-sequence'
            ]

    command = (" ".join(cmd))
    print(command)
    #os.system('/usr/bin/java -version')
    os.system(command)
    #p = Popen(command, stdout=PIPE, stderr=STDOUT)
        
seeds = (1,2,3)
nTopics = 100
# alpha = 50
#    print '# of topics:', nTopics
#    print
if taskHas(0):
    for seed in seeds:
        print ('seed:', seed)
        cmd = [ javapath+'java','-Xmx4G','-ea', '-Dfile.encoding=UTF-8', '-classpath',
                classpath+'/bin:'+classpath+'/lib/mallet-deps.jar', 
                'cc.mallet.topics.tui.Vectors2Topics', 
                '--input', mainDir+'/post.mallet/',
                '--num-threads', '4',
                '--num-iterations', '1000',
                '--show-topics-interval', '2001',
                '--num-topics', str(nTopics),
                #'--alpha', str(alpha),
                '--random-seed', str(seed),
                '--output-topic-keys', mainDir+'/topics/topic-keys-%d-%d.txt' %(nTopics,seed),
                '--word-topic-counts-file', mainDir+'/topics/counts-%d-%d.txt' %(nTopics,seed),
                '--output-doc-topics', mainDir+'/topics/doc-topics-%d-%d.txt' %(nTopics,seed),
                '--output-state', mainDir +'/topics/doc-state-%d-%d.txt' %(nTopics,seed)
        ]
        command = (" ".join(cmd))
        print(command)
        os.system(command)
        #print ' '.join(cmd[11:])

                
if taskHas(1):
    '''
    Read the word-topic-counts file produced by MALLET for each seed
    Count total words for each topic
    Generate topic words for each topic
    Define the distance functions
    '''
    seed_topic_word2ct = []
    seed_topic_totalCt = []
    seed_topic_words = []
    for seed in seeds:
        seed_topic_word2ct.append([])
        seed_topic_totalCt.append([])
        seed_topic_words.append([])
        for j in range(nTopics):
            seed_topic_word2ct[-1].append({})
    for i in range(len(seeds)):
        seed = seeds[i]
        with open(mainDir+'/topics/counts-%d-%d.txt'%(nTopics,seed),'r') as f:
            nWords = 0
            for line in f:
                nWords += 1
                items = line.split()
                word = items[1]
                totalCt = 0
                for t_ct in items[2:]:
                    t,ct = t_ct.split(':')
                    seed_topic_word2ct[i][int(t)][word] = int(ct)   
        for j in range(nTopics):
            word2ct = seed_topic_word2ct[i][j]
            seed_topic_totalCt[i].append(sum(word2ct.values()))
            
    for i in range(len(seeds)):
        for j in range(nTopics):
            word2ct = seed_topic_word2ct[i][j]
            wordSorted = sorted(word2ct,key=lambda w:(-word2ct[w],w))
            seed_topic_words[i].append(wordSorted[:20])

#print(seed_topic_words)
if taskHas(2):
    '''
    Write the top words for each topic and each seed
    Write the count of each word if choose to
    '''
    totalTokenCounts = 0
    show_count = True
    seed_topic_docCounts = [[0]*nTopics for s in seeds]
    seed_topic_alpha = [[0]*nTopics for s in seeds]
    for i in range(len(seeds)):
        #topicSorted = sorted(range(nTopics),key=lambda j:-seed_topic_docCounts[i][j])
        with open(mainDir+'/topics/topics-%d-%d.txt' %(nTopics,seeds[i]), 'w') as f:
            for j in range(nTopics): #topicSorted:
                if show_count:
                    wordCtStr = []
                    #print(seed_topic_words[i][j])
                    if seed_topic_words[i][j] != []:
                        w0 = seed_topic_words[i][j][0]
                        #ct0 = seed_topic_word2ct[i][j][w0]
                        for w in seed_topic_words[i][j]:
                            ct = seed_topic_word2ct[i][j][w]
                            if True: #ct>0.1*ct0:
                                wordCtStr.append(w+':'+str(ct))
                        f.write('%d\t%s\n' %(j,' '.join(wordCtStr)))
                else:
                    f.write('%d\t%s\n' %(j,' '.join(seed_topic_words[i][j])))

def dist_JS(seedpair,topicpair):
    b = 100.0
    s1,s2 = seedpair
    t1,t2 = topicpair
    words1 = set(seed_topic_word2ct[s1][t1])
    words2 = set(seed_topic_word2ct[s2][t2])
    D1 = b*seed_topic_totalCt[s1][t1]+nWords
    D2 = b*seed_topic_totalCt[s2][t2]+nWords
    n = 0
    score = 0
    words = words1|words2
    n = len(words)
    N1,N2 = np.zeros(n),np.zeros(n)
    for i,w in enumerate(words):
        N1[i] = b*seed_topic_word2ct[s1][t1].get(w,0)+1.0
        N2[i] = b*seed_topic_word2ct[s2][t2].get(w,0)+1.0
    p1Inv,p2Inv,avgInv = D1/N1,D2/N2,2*D1*D2/(N1*D2+N2*D1)
    score = np.sum(np.log(avgInv)/avgInv-(np.log(p1Inv)/p1Inv+np.log(p2Inv)/p2Inv)/2)
    p1Inv,p2Inv,avgInv = D1,D2,2*D1*D2/(D2+D1)
    score += (nWords-n)*(log(avgInv)/avgInv-(log(p1Inv)/p1Inv+log(p2Inv)/p2Inv)/2)
    return score/log(2)

dist = dist_JS
distDir = 'dist-JS'
cutoff = 0.7

if taskHas(3):
    '''
    For every pair of sets of topics, compute the pairwise distance of topics
    '''
    spair2tpair2dist = {}
    start = time.time()
    for a in range(len(seeds)):
        for b in range(a+1,len(seeds)):
            print (seeds[a],seeds[b])
            spair2tpair2dist[(a,b)] = {}
            for i in range(nTopics):
                if (i+1)%10==0:
                    print (i+1),
                for j in range(nTopics):

                    d = dist((a,b),(i,j))
                    spair2tpair2dist[(a,b)][(i,j)] = d
            print
    print ('Time on pairs: %.1f min' %((time.time()-start)/60.0))
    
if taskHas(4):
    '''
    Align the 3 sets of topics and 
    find out what triangles are small.
    '''
    triple2maxdist = {}
    for i in range(nTopics):
        if (i+1)%100==0:
            print (i+1)
        for j in range(nTopics):
            d01 = spair2tpair2dist[(0,1)][(i,j)]
            if d01>cutoff:
                continue
            for k in range(nTopics):
                d02 = spair2tpair2dist[(0,2)][(i,k)]
                if d02>cutoff:
                    continue
                d12 = spair2tpair2dist[(1,2)][(j,k)]
                if d12>cutoff:
                    continue
                maxdist = max(d01,d02,d12)
                triple2maxdist[(i,j,k)] = maxdist
    print
    tripleUsed = [set(),set(),set()]
    print ('Sorting triples ...')
    tripleSorted = sorted(triple2maxdist,key=triple2maxdist.get)
    print ('done.')
    triplesAligned = []
    for i,j,k in tripleSorted:
        if i in tripleUsed[0] or j in tripleUsed[1] or k in tripleUsed[2]:
            continue
        triplesAligned.append((i,j,k))
        tripleUsed[0].add(i)
        tripleUsed[1].add(j)
        tripleUsed[2].add(k)
    
if taskHas(5):
    '''
    Write the triples of topics and their merged topics into file.
    '''
    beta = 0.01
    if not os.path.exists(mainDir+distDir):
        os.mkdir(mainDir+distDir)
    with open(mainDir+'/topics/'+distDir+'/topic%d-triples.txt' %nTopics,'w') as f, open(
            mainDir+'/topics/'+distDir+'/topic%d-merged.txt' %nTopics,'w') as g:
        line = 'cutoff = %.3f\n\n' %cutoff
        f.write(line)
        g.write(line)
        for triple in triplesAligned:
            i,j,k = triple
            d01 = spair2tpair2dist[(0,1)][(i,j)]
            d02 = spair2tpair2dist[(0,2)][(i,k)]
            d12 = spair2tpair2dist[(1,2)][(j,k)]
            maxdist = triple2maxdist[triple]
            f.write('(%d,%d,%d): %.2f, %.2f, %.2f\n' %(i,j,k,d01,d02,d12))
            word2triplect = {}
            for m in range(len(seeds)):
                wordCtStrs = []
                for w in seed_topic_words[m][triple[m]]:
                    ct = seed_topic_word2ct[m][triple[m]][w]
                    word2triplect[w] = word2triplect.get(w,0) + ct
                    wordCtStrs.append('%s:%d' %(w,ct))
                f.write('\t%d: %s\n' %(triple[m],' '.join(wordCtStrs[:20])))

            wordSorted = sorted(word2triplect,key=lambda w:(-word2triplect[w],w))
            triTotalCt = seed_topic_totalCt[0][i]+seed_topic_totalCt[1][j]+seed_topic_totalCt[2][k]
            triTotalWt = triTotalCt+3*nWords*beta
            accWeight = 0
            words = []
            inserted = False
            for w in wordSorted[:20]:
                words.append(w)
                accWeight += seed_topic_word2ct[0][i].get(w,0)+seed_topic_word2ct[1][j].get(w,0)+seed_topic_word2ct[2][k].get(w,0)+3*beta
                if not inserted and accWeight > 0.9*triTotalWt:
                    words.append('|')
                    inserted = True
            g.write('(%d,%d,%d): %s\n' %(i,j,k,' '.join(words)))
                
if taskHas(6):
    '''
    Write topics with prevalence
    '''
    seed_topic_docCounts = [[0]*nTopics for s in seeds]
    nDocs = 0
    for i in range(len(seeds)):
        with open(mainDir+'/topics/doc-topics-%d-%d.txt' %(nTopics,seeds[i])) as f:
            for line in f:
                items = line.rstrip().split('\t')
                if i == 0:
                    nDocs += 0.01
                print((items))
                docLen = int(items[0])
                for ttc in items[1].split():
                    topic,tokenCounts = ttc.split(':')
                    print('topic:%s,count:%s',topic,tokenCounts)
                    concentration = float(tokenCounts)/docLen
                    if (docLen>=80 and concentration>=0.05) or (docLen<80 and int(tokenCounts)>=4):
                        seed_topic_docCounts[i][int(topic)] += 1# int(tokenCounts)

    topic_count_triple_str = []
    with open(mainDir+'/topics/'+distDir+'/topic%d-combined.txt' %nTopics) as f:
        for line in f:
            if line.startswith('('):
                tripleStr = line[1:line.index(')')]
                triple = tuple([int(s) for s in tripleStr.split(',')])
                docCounts = [seed_topic_docCounts[i][triple[i]] for i in range(3)]
                docCount = sorted(docCounts)[1]
                maxdist = triple2dist[triple]
                topic_count_triple_str.append((docCount,tripleStr,line[line.index(':')+1:]))
    nStables = len(topic_count_triple_str)
    sortedIdx = sorted(range(nStables),key=lambda k:-topic_count_triple_str[k][0])
    with open(mainDir+'/topics/'+distDir+'/stable-topics-%d(alpha=%d).txt' %(nTopics,alpha),'w') as f:
        f.write('Triple\tPrevalence\tPost\tPubmed\tPubmed/Post\tTopic words\n')
        for i,idx in enumerate(sortedIdx):
            docCount,tripleStr,topicstr = topic_count_triple_str[idx]
            f.write('%s\t%.2f%s\t%s' %(tripleStr.replace(',',';'),docCount/nDocs,'%',topicstr))

if taskHas(7):
    triple2words20 = {}
    with open(mainDir+'/topics/'+distDir+'/topic%d-combined.txt' %nTopics) as f:
        for line in f:
            if line.startswith('('):
                triple = line[1:line.index(')')]
                triple2words20[triple]=line[line.index(':')+1:].split()
    triple2count = {}
    nDocs = 0
    alltriple2word2count = {}

    with open(mainDir+'/topics/'+distDir+'/doc-topic-word-sumcount-%d.txt' %nTopics,'w'
              ) as f, open(mainDir+'/topics/doc-state-%d-%d.txt' %(nTopics,seeds[0])
              ) as g0, open(mainDir+'/topics/doc-state-%d-%d.txt' %(nTopics,seeds[1])
              ) as g1, open(mainDir+'/topics/doc-state-%d-%d.txt' %(nTopics,seeds[2])
              ) as g2:
        for line0 in g0:
            line1 = g1.readline();
            line2 = g2.readline();
            nDocs += 1
            lines = [line0,line1,line2]
            seed_topic2word2count = [{},{},{}]
            docWord2count = {}
            for i in range(len(seeds)):
                line = lines[i]
                for w_t in line.split('\t')[1].split():
                    w,t = w_t.split('^')
                    if i==0:
                        docWord2count[w] = docWord2count.get(w,0)+1
                    if not t in seed_topic2word2count[i]:
                        seed_topic2word2count[i][t] = {}
                    seed_topic2word2count[i][t][w] = seed_topic2word2count[i][t].get(w,0) + 1
            triple2word2count = {}
            for triple in triple2words20:
                tt = triple.split(',')
                nMatchedTopics = 0
                words = set()
                for i in range(len(seeds)):
                    if tt[i] in seed_topic2word2count[i]:
                        words |= set(seed_topic2word2count[i][tt[i]].keys())
                        nMatchedTopics += 1
                words10 = set(triple2words20[triple][:10])
                #words20 = set(triple2words20[triple])
                if nMatchedTopics==0:
                    continue
                for w in words:
                    counts = [0,0,0]
                    for i in range(len(seeds)):
                        counts[i] = seed_topic2word2count[i].get(tt[i],{}).get(w,0)
                    if w in words10:
                        #countSorted = sorted(counts)
                        if not triple in triple2word2count:
                            triple2word2count[triple] = {}
                        triple2word2count[triple][w] = sum(counts)#Sorted)
                    if not triple in alltriple2word2count:
                        alltriple2word2count[triple] = {}
                    alltriple2word2count[triple][w] = alltriple2word2count[triple].get(w,0)+sum(counts)
            triple2weight = {t:sum(triple2word2count[t].values())*len(triple2word2count[t].keys()) for t in triple2word2count}
            tripleSorted = sorted(triple2weight,key=lambda t:-triple2weight[t])
            tripleWordCountStrs = []
            tripleWord2count = {}
            weight2triples = {}
            for triple in tripleSorted:
                word2count = triple2word2count[triple]
                weight = triple2weight[triple]
                if not weight in weight2triples:
                    weight2triples[weight] = []
                weight2triples[weight].append(triple)
                wordCountStrs = []
                for w in word2count:
                    tripleWord2count[w] = tripleWord2count.get(w,0) + word2count[w]
                    wordCountStrs.append('%s:%d' %(w,word2count[w]))
                tripleWordCountStrs.append(triple+'{'+','.join(wordCountStrs)+'}')
            docLen = sum(docWord2count.values())
            f.write(line0.split('\t')[0]+'\t')#+str(docLen)+'\t')#+str(sum(tripleWord2count.values()))+'\t')
            f.write(''.join(tripleWordCountStrs))
#            for w in docWord2count:
#                if nDocs==0:# and 3*docWord2count[w]!=topicWord2count[w]:
#                    print line0.split('\t')[0], w, tripleWord2count.get(w,0), 3*docWord2count[w]
            f.write('\n')
            weightSorted = sorted(weight2triples,reverse=True)
            for weight in weightSorted[:2]:
                if True: #(docLen>=80 and count>=0.05*3*docLen) or (docLen<40 and count>=3*2):
                    for triple in weight2triples[weight]:
                        triple2count[triple] = triple2count.get(triple,0) +1
                    if len(weight2triples[weight])>=2:
                        break
                    
if taskHas(8):
    triple2words20 = {}
    with open(mainDir+'/topics/'+distDir+'/topic%d-combined.txt' %nTopics) as f:
        for line in f:
            if line.startswith('('):
                tripleStr = line[1:line.index(')')]
                triple2words20[tripleStr]=line[line.index(':')+1:]
    tripleSorted = sorted(triple2count,key=lambda t:-triple2count[t])
    with open(mainDir+'/topics/'+distDir+'/stable-topics-%d(alpha=%d)-1.txt' %(nTopics,alpha),'w') as f:
        f.write('Triple\tPrevalence\tTopic words\n')
        for triple in tripleSorted:
            words20 = triple2words20[triple]
            docCount = triple2count.get(triple,0)
            f.write('%s\t%.2f%s\t%s' %(triple.replace(',',';'),docCount*100.0/nDocs,'%',words20))
            
if taskHas(9):
    tripleSorted = sorted(triple2count,key=lambda t:-triple2count[t])
    with open(mainDir+'/topics/'+distDir+'/stable-topics-wordcounts-%d.txt' %nTopics,'w') as f:
        for triple in tripleSorted:
            docCount = triple2count[triple]
            word2count = alltriple2word2count[triple]
            words10 = sorted(word2count,key=lambda w:(-word2count[w],w))
            wordCtStrs = ['%s:%d' %(w,word2count[w]) for w in words10]
            f.write('%s\t%.2f\t%s' %(triple,
                docCount*100.0/nDocs,' '.join(wordCtStrs)))
            f.write('\n')

