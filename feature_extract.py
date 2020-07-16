import os
import pressure
import zones
import pandas as pd
import numpy as np


def extract(directory):
    dataset = pd.DataFrame({'Nama File':[],'Rerata':[],'Persentase':[],'Zona Atas':[],'Zona Tengah':[],'Zona Bawah':[],
        'Tekanan Tulisan':[],'Dominasi Zona':[]})
    try:
        dataCount = 0
        for label in os.listdir(directory):
            for subLabel in os.listdir(os.path.join(directory, label)):
                #label = dominasi zona
                #subLabel = tekanan tulisan
                for imgFile in os.listdir(os.path.join(directory, label, subLabel)):
                    if imgFile.endswith(".jpg") or imgFile.endswith(".jpeg") or imgFile.endswith(".png"):
                        file_name = directory+'/'+label+'/'+subLabel+'/'+imgFile

                        features = [file_name]
                        features += pressure.extract(file_name)
                        features += zones.extract(file_name)

                        new_data = {'Nama File':imgFile,'Rerata':features[1],'Persentase':features[2],'Zona Atas':int(features[3]),
                            'Zona Tengah':int(features[4]),'Zona Bawah':int(features[5]),'Tekanan Tulisan':subLabel,'Dominasi Zona':label}

                        dataset = dataset.append(new_data, ignore_index=True)

                        dataCount += 1
                        print(dataCount, imgFile,"Done")
                    else:
                        continue
                    
    except:
        print("Folder Dataset Tidak Valid!")

    return dataset

#tes = extractFolder(r"D:\DataKuliah\Skripsi\App\Graphology-RSVM\dataset_image")
#print(tes)
#tes.to_csv('test_data.csv', index=False)