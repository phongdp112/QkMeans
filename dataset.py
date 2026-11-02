import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, normalize

class Dataset:
    
    def __init__(self, dataset_name, preprocessing = None):
        self.dataset_name = dataset_name
        self.preprocessing = preprocessing
        self.ground_truth = None
        self.df = None
        self.original_df = self.load_dataset(to_preprocess=False)
        if preprocessing is not None:
            self.df = self.load_dataset(to_preprocess=True)
        else:
            self.df = self.original_df
        self.N = len(self.df.columns) # số chiều của 1 vecto dữ liệu
        self.M = len(self.df)         # số vecto dữ liệu

    def load_dataset(self, to_preprocess = True):
        if self.dataset_name == 'iris':
            self.df = self.load_iris(to_preprocess)
        return self.df
    
    def load_iris(self,to_preprocess):
        df = datasets.load_iris(as_frame=True).frame
        # đổi tên cột
        df.columns = ["f0","f1","f2","f3","class"]
        self.ground_truth = df['class']
        # xóa cột class và cột f1 -> còn cột f0 , f2, f3 
        df = df.drop('class', axis=1)
        df = df.drop('f1', axis=1)
        # đổi tên cột -> f0, f1, f2
        df.columns = ["f0", "f1", "f2"]
        # đưa từng cột của bảng về trung bình 0 và độ lệch chuẩn 1
        df = self.scale(df)

        if to_preprocess:
            df = self.preprocess(df)
        return df

    # Hàm đưa từng cột của bảng về trung bình 0 và độ lệch chuẩn 1 (z-scores)
    def scale(self, data):
        scaler = StandardScaler()                                  # tạo bộ chuẩn hoá, chưa tính gì
        data.loc[:,:] = scaler.fit_transform(data.loc[:,:])        # tính rồi ghi kết quả về bảng
        return data

    def preprocess(self, df):
        if self.preprocessing == 'ISP':
            df = self.ISP(df)
        elif self.preprocessing == '2-norm':
            df = self.L2_norm(df)
        else:
            print("ERROR: WRONG PREPROCESSING")
            exit()
        return df

    def ISP(self, df):
        n = len(df.columns)
        m = len(df)
        self.N = n+1
        column_name = 'f' + str(n)
        df[column_name] = 0.0
        for j in range(m):
            s = sum(df.iloc[j, :]**2)
            for i in range(n):
                df.iloc[j,i] = 2*df.iloc[j,i]
            df.iloc[j,i+1] = s-1
            df.iloc[j, :] = df.iloc[j, :]/(s+1)
        return df

    def L2_norm(self, df):
        df.loc[:,:] = normalize(df.loc[:,:])
        return df
