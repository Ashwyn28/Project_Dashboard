import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from data_processing.normalise import scale 

# PCA feature fusion

def pca(data, flabels):

    data = data.iloc[0:16]

    pca = PCA()
    pca.fit(data)
    pca_data = pca.transform(data)
    
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    pca_df = pd.DataFrame(pca_data, columns=labels)

    # adding pca labels to dataset

    pca_df_with_labels = pca_df.join(flabels)

    data = pca_df_with_labels

    return data



