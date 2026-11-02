import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from QRAM import encodeVector
import pandas as pd
import numpy as np

class QKmeans:
    def __init__(self,dataset,conf):
        self.dataset = dataset
        self.N = self.dataset.N  # số chiều của vecto dữ liệu 
        self.M = self.dataset.M   # số bản ghi dữ liệu 
        self.sc_tresh = conf['sc_tresh']
        self.centroids = None
        self.old_centroids = None
        self.quantization = conf['quantization']
        self.cluster_assignment = [0]*self.M
        self.shots = conf['shots']
        self.K = conf['K']
    
    def Qk1_1(self, provider = None, check_prob = False):
        distances = [] 
        N = self.N
        Rqram_qbits = 1                        # qubit thanh ghi r
        Aknn_qbits = 1                         # qubit ancilla a
        I_qbits = math.ceil(math.log(N,2))     # qubit đánh chỉ số N đặc trưng, n̂ = ⌈log2 N⌉ trong bài báo
        Aqram_qbits = I_qbits + Aknn_qbits - 2 # qubit ancilla phụ cho mcry
        self.max_qbits = I_qbits + Aknn_qbits + Rqram_qbits + Aqram_qbits
        a = QuantumRegister(1, 'a')            # ancilla — phân biệt nhánh record/centroid
        i = QuantumRegister(I_qbits, 'i')      # chỉ số đặc trưng
        r = QuantumRegister(1, 'r')            # nơi mã hoá giá trị
        if Aqram_qbits > 0:
            q = QuantumRegister(Aqram_qbits, 'q')  # ancilla phụ

        outcome = ClassicalRegister(2, 'bit')  # 2 bit cổ điển để đo
        for index_v, vector in enumerate(self.dataset.df.values):
            centroid_distances = []
            for index_c in range(len(self.centroids)):
                if Aqram_qbits > 0:
                    circuit = QuantumCircuit(a, i, r, q, outcome)
                else:
                    circuit = QuantumCircuit(a, i, r, outcome)
                    q = None
                circuit.h(a)
                circuit.h(i)

                encodeVector(circuit, self.centroids[index_c], i, a[:]+i[:], r[0], q)  # (1) mã hoá CENTROID
                circuit.x(a)
                encodeVector(circuit, vector, i, a[:]+i[:], r[0], q)                    # (2) mã hoá RECORD 
                circuit.measure(r,outcome[0])
    
                circuit.h(a)
    
                circuit.measure(a,outcome[1])
    
                shots = self.shots
                
                simulator = AerSimulator()
                #simulator.set_options(device='GPU')
                job = simulator.run(transpile(circuit, simulator), shots=shots)
                result = job.result()
                counts = result.get_counts(circuit)

                goodCounts = {k: counts[k] for k in counts.keys() & {'01','11'}}

                try:
                    n_p0 = goodCounts['01']
                except:
                    n_p0 = 0
                
                euclidian = 4-4*(n_p0/sum(goodCounts.values()))
                euclidian = math.sqrt(euclidian)
                if self.dataset.preprocessing == '2-norm':
                    r_norm = self.records_norm[index_v]
                    c_norm = self.centroids_norm[index_c] 
                    euclidean_2 = math.sqrt((r_norm-c_norm)**2+(euclidian**2)*(r_norm*c_norm))
                    euclidian = euclidean_2

                centroid_distances.append(euclidian)

            distances.append(centroid_distances)

        self.cluster_assignment = [(i.index(min(i))) for i in distances] # for each vector takes the closest centroid to perform the assignemnt

    def computing_centroids(self):
        series = []
        old_series = []
        index_centroids = []
        index_oldcentroids = []
        for i in range(self.K):
            if i in self.cluster_assignment:
                series.append(self.dataset.original_df.loc[[index for index, n in enumerate(self.cluster_assignment) if n == i]].mean())
                index_centroids.append(i)
            else:
                old_centroid = pd.Series(self.centroids[i])
                old_centroid = old_centroid.rename(lambda x: "f" + str(x))
                old_series.append(old_centroid)
                index_oldcentroids.append(i)
                
        if len(series)>0:
            df_centroids = pd.concat(series, axis=1).T
            df_centroids.index = index_centroids  
        else:
            df_centroids = pd.DataFrame(columns=self.dataset.original_df.columns)
        if len(old_series)>0:
            df_oldcentroids = pd.concat(old_series, axis=1).T
            df_oldcentroids.index = index_oldcentroids
        else:
            df_oldcentroids = pd.DataFrame(columns=self.dataset.df.columns)
        df_centroids = self.dataset.preprocess(df_centroids)
        df_centroids = pd.concat([df_oldcentroids, df_centroids], axis=0, sort=False).sort_index()
        self.centroids_norm = df_centroids.apply(lambda row: np.linalg.norm(row), axis=1).tolist()

        self.centroids = df_centroids.values    

    def stop_condition(self):
        if self.old_centroids is None:
            return False

        if np.linalg.norm(self.centroids-self.old_centroids, ord='fro') >= self.sc_tresh:
            return False
        
        return True
    
    def run(self,  initial_centroids=None, seed = 123, real_hw = False):
        if initial_centroids is None:
            df_centroids = self.data.sample(n=self.K, random_state=seed)
        else:
            df_centroids = pd.DataFrame(initial_centroids, columns=self.dataset.original_df.columns)
        self.centroids_norm = df_centroids.apply(lambda row: np.linalg.norm(row), axis=1).tolist()
        df_centroids = self.dataset.preprocess(df_centroids)

        self.centroids = df_centroids.values        


        while not self.stop_condition():
            self.old_centroids = self.centroids.copy()
            if self.quantization == 1:
                self.Qk1_1()
            else: 
                self.Qk1_1()
            self.computing_centroids()

