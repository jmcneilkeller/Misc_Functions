from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, ward, linkage, cophenet

def dendogram(df,linkage='ward',distance='euclidean'): 
    # Takes in a dataframe, cluster linkage and distance and produces a dendogram + cophenet score.
    Z = shc.linkage(df,linkage,distance)
    # Check cophenet correlation score.
    c, coph_dists = cophenet(Z, pdist(df))
    print('Cophenet Correlation:',c)
    
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('No. of clusters')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()
    
    #Shows the distance of the 12 final splits.
    print('Last 12 cluster distances:',Z[-12:,2])

def agg_clust(df,n,affinity,linkage):
    # Performs agglomerative clustering based on results of best params.
    # Returns clusters a list of assigned clusters for each data point.
    agg_clust = AgglomerativeClustering(n_clusters=n,affinity=affinity,linkage=linkage)
    assigned_clust = agg_clust.fit_predict(df)
    return agg_clust,assigned_clust