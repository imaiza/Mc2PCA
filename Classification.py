# Classification of the new MTS based on the distance to the centroids
# If we want to classify various MTS, apply the function several times

def cluster_prediction(df_new, proj_centroids):
    
    # df_new = DataFrame containing the new MTS to classify
    # proj_centroids = centroids of the model (output of Mc2PCA)
    
    lista_colada = df_new['Fichero'].unique()
    num_clust = len(proj_centroids)
    error_array = np.zeros([num_clust, len(lista_colada)]) 
    
    for c_ind in range(num_clust):
            
        proj = proj_centroids[c_ind] # We call the prototype/centroid
        sst = np.matmul(np.asarray(proj), proj.T) # np.matmul: matrix product of two arrays
            
        # Calculate the error in all clusters:
        for (p_ind, colada) in enumerate(lista_colada):
            print(colada)
            error_array[c_ind, p_ind] = error_calculation(sst, df.loc[df['Fichero'] == colada])
        
                        
    temp_attribution = np.argmin(error_array, 0) # Returns the indices of the minimum values along an axis
    print('The MTS is assigned to the cluster ', temp_attribution)
    
    return temp_attribution, error_array