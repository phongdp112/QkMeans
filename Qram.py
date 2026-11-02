import numpy as np
def indexing(circuit, Qregister, index):
    size = Qregister.size
    xored = index ^ (pow(2, size) - 1)
    j=1
    for k in (2**p for p in range(0, size)):
        if xored & k >= j:
            circuit.x(Qregister[j-1])
        j = j+1

def encodeVector(circuit, data, i, controls, rotaionQubits, ancillaQubits):
