import h5py
import csv
import pandas as pd
data=pd.read_hdf('./data/s1_CP_smoothPursuitData.h5')
print(data.head())
