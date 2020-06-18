import os
import pressure
import zones
import pandas as pd
import numpy as np

def test(directory):
    dataset = pd.DataFrame({'Nama File':[],'Rerata':[],'Persentase':[],'Zona Atas':[],'Zona Tengah':[],'Zona Bawah':[],
        'Tekanan Tulisan':[],'Dominasi Zona':[]})

    for label in os.listdir(directory):
        if label.endswith(".jpg") or label.endswith(".jpeg") or label.endswith(".png"):
            file_name = directory+'/'+label

            features = [file_name]
            features += pressure.start(file_name)
            features += zones.start(file_name)

            new_data = {'Nama File':label,'Rerata':features[1],'Persentase':features[2],'Zona Atas':int(features[4]),
                'Zona Tengah':int(features[5]),'Zona Bawah':int(features[6]),'Tekanan Tulisan':features[3],'Dominasi Zona':features[7]}

            dataset = dataset.append(new_data, ignore_index=True)
        else:
            continue
    
    return dataset

