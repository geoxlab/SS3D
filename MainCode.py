# -*- coding: utf-8 -*-

import networkx as nx 
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import itertools 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import asyn_fluidc
from networkx.algorithms import community


#Construct edges
def ConstructEdges(lineId, topos):
    """to construct a edge list according to the topological relations between lines
    Args:
        lineId:a list of line id
        topos: given a line, a list of ids of interacted lines
    return:
        a list of edges
    """
    edgelist = []
    for i in range(0,len(topos)):
        templine = lineId[i]
        idx1 = lineId.index(templine) #find the index of current line id
        temptopo = topos[i].split(',')
        for j in range(0,len(temptopo)):
            #templist = [] #a pair of line ids, coresponds to one edge            
            topoline = temptopo[j]
            idx2 = lineId.index(int(topoline))
            templist = [idx1,idx2]
            templist.sort()
            edgelist.append(templist)
    return edgelist
            
def GetClusterID(Clist):
    NodeID = []
    IDList = []
    for i in range(len(Clist)):
        for j in range(len(Clist[i])):
            IDList.append(i)
            NodeID.append(Clist[i][j])
            
    return NodeID, IDList
        
    
            
        

if __name__=="__main__":

    #read data from csv

    csvfile='Campus_axial_flow.csv'
    lines=pd.read_csv(csvfile)

####read the attributes
    NodeId = list(lines.index)
    lineId = lines['Id'].tolist()
    topos = lines['ID_Total'].tolist()
    edgelist = ConstructEdges(lineId, topos)
    edges = []
    [edges.append(i) for i in edgelist if not i in edges] #remove the duplicate edges
    flow0303 = np.array(lines['Flow0303'])
    flow0304 = np.array(lines['Flow0304'])
    flow0305 = np.array(lines['Flow0305'])
    flow0306 = np.array(lines['Flow0306'])
    flow0307 = np.array(lines['Flow0307'])
    flow0308 = np.array(lines['Flow0308'])
    flow0309 = np.array(lines['Flow0309'])
    flow0310 = np.array(lines['Flow0310'])
    flow = flow0304 + flow0305+ flow0306+ flow0307+ flow0308+ flow0309+ flow0310

##space syntax measures
    Length = np.array(lines['Shape_Leng'])
    Control = np.array(lines['Control'])
    MeanDepth = np.array(lines['MeanDepth'])
    GInteg = np.array(lines['GInteg'])
    LInteg = np.array(lines['LInteg'])
    TotalDepth = np.array(lines['TotalDepth'])
    LocalDepth = np.array(lines['LocalDepth'])
#    
#    
#    #generate graph
    G = nx.Graph()
    G.add_nodes_from(NodeId)
    G.add_edges_from(edges)
    print(G.number_of_nodes())
    print(G.number_of_edges())
#    nx.draw(G)
#    plt.savefig("graph.png")
    
####Calculate centrality indicators
    degree = nx.degree_centrality(G)
#    deg = G.degree
    eigenvector =  nx.eigenvector_centrality(G,max_iter=500)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
#    
    d_centrality = list(degree.values())
    e_centrality = list(eigenvector.values())
    c_centrality = list(closeness.values())
    b_centrality = list(betweenness.values())
    p_centrality = list(pagerank.values())

#####calculate the correlation coefficients between flow and metrics    
    p1 = np.corrcoef(flow,np.array(Control))
    p2 = np.corrcoef(flow,np.array(MeanDepth))
    p3 = np.corrcoef(flow,np.array(GInteg))
    p4 = np.corrcoef(flow,np.array(LInteg))
    p5 = np.corrcoef(flow,np.array(TotalDepth))    
    p6 = np.corrcoef(flow,np.array(LocalDepth))
    p7 = np.corrcoef(flow,np.array(d_centrality))
    p8 = np.corrcoef(flow,np.array(e_centrality))
    p9 = np.corrcoef(flow,np.array(c_centrality))
    p10 = np.corrcoef(flow,np.array(b_centrality))
    p11 = np.corrcoef(flow,np.array(p_centrality))
    p12 = np.corrcoef(flow,np.array(Length))

#####export the flow and metrics to a csv document    
    Data = pd.DataFrame()
    Data['ID']=lineId
    Data['flow0304']=flow0304
    Data['flow0305']=flow0305
    Data['flow0306']=flow0306
    Data['flow0307']=flow0307
    Data['flow0308']=flow0308
    Data['flow0309']=flow0309
    Data['flow0310']=flow0310
    Data['TotalFlow']=flow
    Data['Length']=Length
    Data['degree']=d_centrality
    Data['eigenvector']=e_centrality
    Data['closeness']=c_centrality
    Data['betweenness']=b_centrality
    Data['pagerank']=p_centrality
    Data['Control']=Control
    Data['MeanDepth']=MeanDepth
    Data['GInteg']=GInteg
    Data['LInteg']=LInteg
    Data['TotalDepth']=TotalDepth
    Data['LocalDepth']=LocalDepth
    pickle.dump(Data, open(r'Data.d', 'wb'))
    Data.to_csv('Results.csv')
    data = pickle.load(open(r'Data.d','rb'))

#####Community detection    
#    communities1 = list(greedy_modularity_communities(G))
#    communities2 = list(label_propagation_communities(G))
##    communities3 = list(girvan_newman(G))
#    communities4 = list(asyn_fluidc(G, 5, max_iter=100, seed=1))

#    k = len(communities4)
#    Clist = []
#    for communities in itertools.islice(communities4, k):
##        print(tuple(c for c in communities))
#        Clist.append(tuple(c for c in communities))
#    NodeID, IDList = GetClusterID(Clist)
    
#    # Find modularity
#    part = list(k_clique_communities(G,2))    
#    # Plot, color nodes using community structure
#    values = [part.get(node) for node in G.nodes()]
#    nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
#    plt.show()
    
#    Data = pd.DataFrame()
##    Data['ID']=lineId
#    Data['ID'] = IDList
##    Data['Degree_C']=d_centrality
##    Data['Eigenvector_C']=e_centrality
##    Data['Closeness_C']=c_centrality
##    Data['Betweenness_C']=b_centrality
##    Data['Pagerank_C']=p_centrality
#    Data['CommunityID'] = NodeID
#    Data.to_csv('Community.csv')
    print('OK!')

