def Mc2PCA(df, num_clust, p, num_iter):
    
    '''
    Implementation of the Mc2PCA algorithm proposed by Li[2018]. I included a modification, in order to avoid the existence of 
    empty clusters (something that can occur when working on anomaly detection or when considering a small value of "p" and a 
    high value of "num_clust").
    
    INPUTS:
    
    - df: Dataframe containing all the data, with a column called "Fichero" that provides info about the MTS to which the instance belongs.
    - num_clust: number of clusters.
    - p: number of retained components. 
    - num_inter: max number of iterations.
  
    OUTPUTS:
    
    - colada_classif: dictionary with the final clustering result
    - proj_centroids: centroids of the final clusters
    - error_array: matrix of the errors (of all MTS on all clusters)
 
    '''
        
    # First we create the covariance matrices dictionary
    dict_cov = create_dic_cov(df)
    
    # Return random integers from low (inclusive) to high (exclusive) as a list
    num_int = np.random.randint(low = 0, high = num_clust, size=len(df['Fichero'].unique()), dtype='l')
    
    # Initialize centroid
    centroid_init = []

    for f in range(0, num_clust):
        centroid_init.append([])
    
    list_coladas = df['Fichero'].unique()
    
    # Assign each MTS to a cluster randomly
    for f in range(0, num_clust):
        centroid_init[f] = list_coladas[np.where(num_int == f)] # Lista of arrays
    
    # Create prototype/centroid of each cluster
    proj_centroids = {}
    
    for (i, f) in enumerate(centroid_init):
        proj_centroids[i] = projection_cluster_cov(dict_cov, f, p) # f = list_colada, p = numb_dim
    
    # Define some variables:
    colada_classif = {}
    it = 0
    old_error = 1e32
    total_error = 1e30
    
    # Allocate MTS in the best cluster
    
    while (old_error - total_error) / old_error > 0.0001 and (it < num_iter):
        
        old_error = total_error
        error_array = np.zeros([num_clust, len(list_coladas)]) # Matrix dim: num_clust x 100 (coladas)
        
        for c_ind in range(num_clust):
            
            # Ec.6 of paper (right side)
            proj = proj_centroids[c_ind] # Get prototype/centroid
            sst = np.matmul(np.asarray(proj), proj.T) # np.matmul: matrix product of two arrays
            
            # Calculate error:
            for (p_ind, colada) in enumerate(list_coladas):
                error_array[c_ind, p_ind] = error_calculation(sst, df.loc[df['Fichero'] == colada])
        # print(error_array)
                
                
        temp_attribution = np.argmin(error_array, 0) # Returns the indices of the minimum values along an axis
        print('Cluster attribution:  ', temp_attribution)
        total_error = np.sum(np.min(error_array, 0))
        print("Treating iteration ", it, "Error is ", total_error, " Error diff is ",(old_error - total_error) )
        
        fin = False
        if (old_error - total_error) < 0:
            fin = True
            
        # If the previous error was bigger, we reassign clusters:
        
        if total_error < old_error:
            
            for c_ind in range(num_clust):
                
                print(c_ind)
                colada_clust = list_coladas[np.where(temp_attribution == c_ind)]
                print(colada_clust)
                
                if  colada_clust.size == 0:
                    
                    continue
                    
                else:
                    
                    proj_centroids[c_ind] = projection_cluster_cov(dict_cov, colada_clust, p)
                    colada_classif[c_ind] = colada_clust
        
        ##########################################################################################
        #############     MY MODIFICATION. CREATED TO AVOID EMPTY CLUSTERS   #####################
        ##########################################################################################
        
        # In best scenario (all clusters with at least one MTS) not needed. 
        # Method does not work if nº empty clusters > nº of non empty clusters.
        
        filled_clusters = filled_clusters_calculation(temp_attribution)
        bad = False
        
        while len(set(temp_attribution)) != num_clust:
            
            if fin:
                break
                
            if (len(set(temp_attribution))) < (num_clust/2) :
                print('So many empty clusters!')
                bad = True
                break
            
            # Calculate the biggest error
                
            worst_errors = worst_error_calculator(error = error_array, filled = filled_clusters)
            
            for i in range(num_clust):
                print(i) 

                if i in temp_attribution:
                    
                    continue
                else:
                
                    # Empty cluster
 
                    index = np.where(error_array == worst_errors[0])
                    print('index', index,'y', index[1])
                    
                    # Dont use that MTS again:
                    for t in range(num_clust):
                        
                        error_array[t, index[1]] = 0
                        
                    worst_errors = worst_error_calculator(error = error_array, filled = filled_clusters)                    
                    
                    temp_attribution[index[1]] = i
                    
                    print('New Cluster attribution:  ', temp_attribution)
                    filled_clusters = filled_clusters_calculation(temp_attribution)
                    
                    
            for c_ind in range(num_clust):
                
                colada_clust = list_coladas[np.where(temp_attribution == c_ind)]
                
                colada_classif[c_ind] = colada_clust
            
        if bad:
            print('Process ended due to bad initialization or bad initial conditions (too much clusters ...')
            break
            
        it += 1
            
    return proj_centroids, colada_classif, error_array


# Usage example:

centroid, classif, error = Mc2PCA(df, num_clust = 3, p = 10, num_iter = 50)




# ALL THE FUNCTIONS:



# Creates the covariance matrices:

def cov_colada(df, colada, norm=True):
    
    mts = df.loc[df['Fichero'] == colada]
    mts.drop('Fichero', axis = 1, inplace = True)
    
    # Normalization as stated in the paper
    mts_norm = mts - mts.mean()
    
    if norm:
        cov = np.cov(np.asarray(mts_norm).T)
    else:
        cov = np.cov(np.asarray(mts).T)
        
    return cov


# Create a dict with all the covariance matrices:

def create_dic_cov(df):
    
    dict_cov = {}
    
    list_coladas = df['Fichero'].unique()
    
    for p in list_coladas:
        dict_cov[p] = np.nan_to_num(cov_colada(df, p, norm=True)) # Replace NaN with zero and infinity with large finite numbers 
        
    return dict_cov


# Obtain the common space, performing SVD on the mean of all cov matrices:

def projection_cluster_cov(dict_cov, colada_list, numb_dim):
    
    cov_fin = np.zeros(dict_cov[colada_list[0]].shape)
   
    for colada in colada_list:
        
        cov_temp = dict_cov[colada]
        cov_fin += cov_temp
        
    cov_fin /= len(colada_list)
    
    u, s, v = np.linalg.svd(cov_fin)
    #     print(u.shape,s.shape,v.shape)
    #     print(np.asarray(u)[:,:numb_dim].shape, u.shape,numb_dim)
    
    u_array = np.asarray(u)
    #     print(u_array.shape, numb_dim, 'u_array shape')
    
    return u_array[:, :numb_dim]

# Calculate the error after projecting and reconstructing the MTS:
    
def error_calculation(sst, df):
    
    # sst is defined as sst = np.matmul(np.asarray(proj), proj.T) 
    
    array_new = df.drop('Fichero', axis = 1)
    
    projected = np.matmul(np.asarray(array_new), sst) # Reconstruction
    
    error = np.sum(np.square(np.asarray(array_new - projected))) # Reconstruction error. Dif( Reconstruction - Original )
   
    return error


# Returns list with the index of the filled clusters:

def filled_clusters_calculation(temp_idx):
    
    filled = []
    
    for i in (set(temp_idx)):
        
        if list(temp_idx).count(i) >= 2:
            filled.append(i)
        else:
            continue
            
    return filled

# Calculates the biggest(so worst) error of the filled clusters:

def worst_error_calculator(error, filled):
    
    err_max = np.max(error,1)
    worst_errors = []
            
    for j in range(len(filled)):
    
        maximo = err_max[filled[j]]
        worst_errors.append(maximo)
               
    worst_errors.sort(reverse = True)
    
    return worst_errors

# Calculate the within error cluster:
# Can be used to calculate the BIC and define the optimal number of clusters for the problem:

def within_cluster_error_calculation(df, classif, num_dim):
    
    list_wss = []
    for f in classif.keys():
        
        #print('Taking care of cluster', f)
        proj = projection_cluster_cov(dict_cov, classif[f], num_dim)
        sst = np.matmul(np.asarray(proj), proj.T)
        
        error = 0
        for p in classif[f]:
            error +=  error_calculation(sst, df.loc[df['Fichero'] == p])
            
        list_wss.append(error/len(classif[f]))
        
    return list_wss


