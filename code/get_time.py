#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:42:36 2023

@author: seoeunyang
"""

#Load separated parcellation
# https://nilearn.github.io/dev/modules/datasets.html
# https://nilearn.github.io/dev/auto_examples/03_connectivity/plot_probabilistic_atlas_extraction.html

#atlas2 = datasets.fetch_coords_dosenbach_2010() #  160 labels, 'rois', 'labels', 'networks', 'description'
#atlas3 = datasets.fetch_coords_power_2011() # 264 labels, 'rois', 'description'
#atlas4 = datasets.fetch_coords_seitzman_2018() # 300 labels, 'rois', 'radius', 'networks', 'regions', 'description'

import os
import numpy as np
import pickle 
from nilearn import input_data
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn import image as nimg

path='/Users/seoeunyang/Dropbox/AOMIC/data/'
os.chdir(path)
atlas = datasets.fetch_atlas_aal() #116 labels, 'description', 'maps', 'labels', 'indices'
atlas2 = datasets.fetch_atlas_destrieux_2009() #151 labels, 'maps', 'labels', 'description'
  
# extracts the signal on regions defined via AAl atlas for SPM 12, to construct a functional connectome.

def extract_timeseries(atlas,file_nii):
    func_img = nimg.load_img(file_nii)
    masker = input_data.NiftiLabelsMasker(labels_img=atlas["maps"],
                                          standardize=True,
                                          memory='nilearn_cache',
                                          verbose=1,
                                          detrend=True,
                                         low_pass = 0.08,
                                         high_pass = 0.009,
                                         t_r=2)
    
    cleaned_and_averaged_time_series = masker.fit_transform(func_img) 
    return cleaned_and_averaged_time_series
 
#TimeData={}    
empty=[]
TimeData=pickle.load(open('../Preprocessing/TimeData_aal_116.pkl','rb'))
for i in range(777,929):
    if i not in np.array(list(TimeData.keys())):
        file_nii='sub-'+str(i).zfill(4)+'/func/swarsub-'+str(i).zfill(4)+'-func-sub-'+str(i).zfill(4)+'_task-moviewatching_bold.nii'
        try:
            TimeData[i] = extract_timeseries(atlas,file_nii) 
        except: 
            empty.append(i)
            pass
pickle.dump(TimeData,open('../Preprocessing/TimeData_aal_116.pkl','wb'))

 
#TimeData2={}    
empty2=[]
TimeData2=pickle.load(open('../Preprocessing/TimeData_fetch_atlas_destrieux_151.pkl','rb'))
for i in range(777,929):
    if i not in np.array(list(TimeData2.keys())):
        file_nii='sub-'+str(i).zfill(4)+'/func/swarsub-'+str(i).zfill(4)+'-func-sub-'+str(i).zfill(4)+'_task-moviewatching_bold.nii'
        try:
            TimeData2[i] = extract_timeseries(atlas2,file_nii) 
        except: 
            empty2.append(i)
            pass

pickle.dump(TimeData2,open('../Preprocessing/TimeData_fetch_atlas_destrieux_151.pkl','wb'))


######################
#### load dataset ####
######################
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

TimeData = pickle.load(open('TimeData_aal_116.pkl','rb')) #736,290,116
correlation_measure = ConnectivityMeasure(kind="correlation")
correlation_matrix = correlation_measure.fit_transform([TimeData[1]])[0]

# Display the correlation matrix  
np.fill_diagonal(correlation_matrix, 0) # Mask out the major diagonal
plotting.plot_matrix(
    correlation_matrix, labels=atlas['labels'], colorbar=True, vmax=1, vmin=-1, figure=[12.4, 10.8]
) 
plt.savefig('cor_mat_1.png',dpi=300,bbox_inches='tight')



###################### 
##### Connectome #####
###################### 
from nilearn import plotting

coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(
    correlation_matrix, coords, edge_threshold="80%", colorbar=True
)

plotting.show()


#############################################
##### 3D visualization in a web browser #####
#############################################
view = plotting.view_connectome(
    correlation_matrix, coords, edge_threshold="80%"
)

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view
 

 


