#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:25:32 2017

@author: zaheerbabar
"""
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_file
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import networkx.drawing as ng
from sklearn.neighbors import NearestNeighbors


def NN(Embed,words,word,review):
    nn=5
    ind=np.argwhere(words==word)[0][0]
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(np.array(Embed))
    distances, indices = nbrs.kneighbors(np.array(Embed))


    cosineScores=cosine_similarity(Embed, Y=None, dense_output=True)
    cosineScores[cosineScores<0]=0
    df=pd.DataFrame(cosineScores)
    cosine_mat=df.iloc[indices[ind],indices[ind]]
    labels=words[indices[ind]]
    print(review,':',words[indices[ind]])

    G = nx.DiGraph(cosine_mat.values)

    #nx.draw(G)
    #plt.show()
    #print(cosineScores[:,0:10])
    #return labels

def NN_visualize(Embed,words):


    word='molecular'


    review='Orignal'
    review1='First Index'
    review2='second Index'
    review3='third Index'

    labels1=NN(Embed.iloc[:,0:128],words,word,review)
    # pos=nx.spring_layout(Orignal)
    # plt.subplot(221)
    # nx.draw_networkx_labels(Orignal,pos=pos, labels=labels1)
    # nx.draw(Orignal)


    labels2=NN(Embed.iloc[:,10:128], words, word, review1)
    # pos = nx.spring_layout(First)
    # plt.subplot(222)
    # nx.draw_networkx_labels(First, pos=pos, labels=labels2)
    # nx.draw(First)
    #
    labels3=NN(Embed.iloc[:,20:128], words, word, review2)
    # pos = nx.spring_layout(Second)
    # plt.subplot(223)
    # nx.draw_networkx_labels(Second, pos=pos, labels=labels2)
    # nx.draw(Second)
    #
    labels4=NN(Embed.iloc[:,30:128], words, word, review3)
    # pos = nx.spring_layout(Third)
    # plt.subplot(224)
    # nx.draw_networkx_labels(Third, pos=pos, labels=labels4)
    # nx.draw(Third)


    #plt.show()

def main():
    Data = pd.read_csv("Embed_words.csv", encoding='ISO-8859-1')

    Embed_mat = Data.iloc[:, 0:128]
    # Embed_mat.plot(legend=False)
    # plt.plot(np.array(Embed_mat))
    # plt.show
    vocab = Data.iloc[:, 128]
    # Embed_mat=Data.set_index('UNK')
    # Embed_correlationMatrix(Embed_mat,vocab)
    # Scipy_Hierar_clusterAndProcess(Embed_mat,vocab)
    #Kmean_clusterAndProcess(Embed_mat, vocab)
    NN_visualize(Embed_mat,vocab)
    # d=np.random.rand(10,10)
    # G=nx.DiGraph(d)
    # nx.draw(G)
    # plt.show()
    # Data=Data.sort_values("clusterNr",ascending=True)
    # Data.to_csv("DataFileHier.csv",index=False)
    return Data


if __name__ == '__main__':
    Data = main()
