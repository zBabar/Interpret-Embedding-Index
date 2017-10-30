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

def NearestNeigbors(Embed,words):
    cosineScores=cosine_similarity(Embed, Y=None, dense_output=True)
    cosineScores[cosineScores<0]=0
    df=pd.DataFrame(cosineScores,columns=np.array(words))
    df.to_csv('cosine.csv')
    #G = nx.DiGraph(df.values)

    #nx.draw(G)
    #plt.show()
    #print(cosineScores[:,0:10])


def tsne_model():
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)

    return tsne_model


def Kmean_clusterAndProcess(Embed, words):
    vz = np.array(Embed)
    num_clusters = 30
    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                                   init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
    kmeans = kmeans_model.fit(vz)
    kmeans_clusters = kmeans.predict(vz)
    kmeans_distances = kmeans.transform(vz)
    tsne_op = tsne_model()
    tsne_kmeans = tsne_op.fit_transform(kmeans_distances)
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters
    kmeans_df['description'] = words
    output_file("log_lines.html")
    colormap,plot_kmeans=plot_specs()

    plot_kmeans.scatter(kmeans_df['x'], y=kmeans_df['y'],
                         color=colormap[kmeans_clusters])
                        # source=kmeans_df)
    hover = plot_kmeans.select(dict(type=HoverTool))
    hover.tooltips = {"description": "@description", "cluster": "@cluster"}
    show(plot_kmeans)
    #print(tsne_values,tsne_values.shape, kmeans_clusters.shape)
    Data = pd.concat([Embed, words, pd.DataFrame(kmeans.labels_)], axis=1)

    return Data
def plot_specs():
    colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
                         "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053",
                         "#5e9981",
                         "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce",
                         "#d07d3c",
                         "#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])

    plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the Embeddings",
                            tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                            x_axis_type=None, y_axis_type=None, min_border=1)
    return colormap,plot_kmeans
def Hierar_clusterAndProcess(Embed, words):
    Hierar = AgglomerativeClustering(n_clusters=100, linkage='ward').fit(Embed)
    clusters = pd.DataFrame(Hierar.labels_, columns=["clusterNr"])
    Embed = pd.DataFrame(Embed)
    Data = pd.concat([Embed, words, clusters], axis=1)

    return Data


def Scipy_Hierar_clusterAndProcess(Embed, words):
    dist = pdist(Embed)
    Hierar = linkage(dist, 'ward')
    plt.figure()
    # dendrogram(Hierar,truncate_mode='lastp',labels=words,p=100,leaf_rotation=90.,
    # leaf_font_size=12.)
    dendrogram(Hierar, labels=np.array(words), leaf_rotation=90., leaf_font_size=12.)
    plt.show()
    print(len(dist))
    # clusters=pd.DataFrame(Hierar.labels_,columns=["clusterNr"])
    # Embed=pd.DataFrame(Embed)
    # Data=pd.concat([Embed,words,clusters],axis=1)

    # return Data


def Embed_correlationMatrix(Embed, words):
    corrMat = Embed.iloc[0:40, :].T.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corrMat, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 40, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(np.array(words), rotation=90)
    ax.set_yticklabels(np.array(words))
    plt.show()


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
    NearestNeigbors(Embed_mat,vocab)

    # Data=Data.sort_values("clusterNr",ascending=True)
    # Data.to_csv("DataFileHier.csv",index=False)
    return Data


if __name__ == '__main__':
    Data = main()
