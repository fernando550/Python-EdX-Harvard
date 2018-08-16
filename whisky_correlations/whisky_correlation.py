# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:11:08 2018

@author: fnarbona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering

whisky = pd.read_csv('whiskies.txt')
whisky["Region"] = pd.read_csv("regions.txt")

# flavor feature columns from whisky data frame
flavors = whisky.iloc[:,2:14]
corr_flavors = pd.DataFrame.corr(flavors)
corr_whisky = pd.DataFrame.corr(flavors.transpose())

# flavors to whisky plot
#plt.figure(figsize=(10,10))
#plt.pcolor(corr_flavors)
#plt.colorbar()
#
# whisky to flavors plot
#plt.figure(figsize=(10,10))
#plt.pcolor(corr_whisky)
#plt.colorbar()

model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)

whisky["Group"] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)

correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())
correlations = np.array(correlations)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")