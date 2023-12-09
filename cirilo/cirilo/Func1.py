import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from blabla import Algorithms, Funcs
from scipy import stats #Pra calcular a moda 

#função 1 -> mínimo
#################################################
lowerBound = -100
upperBound = 100
x = np.linspace(lowerBound, upperBound, 100)
y = np.linspace(lowerBound, upperBound, 100)

# Crie uma grade de pontos (x, y)
X, Y = np.meshgrid(x, y)

f = Funcs()
alg = Algorithms()
# Calcule os valores da função para cada ponto na grade
Z = f.func_1(X, Y)

# Crie a figura
fig = plt.figure()

# Adicione um subplot 3D
ax = fig.add_subplot(111, projection='3d')

# Plote a superfície 3D
ax.plot_surface(X, Y, Z, cmap='viridis')

# Adicione rótulos aos eixos
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')

# Adicione um título
ax.set_title('Gráfico tridimensional da função')
#########################################################


valoresFunc1HC = []
valoresFunc1LRS = []
valoresFunc1GRS = []
valoresFunc1SAS = []

#calcular o stats.mode(valoresFunc1)

i = 0
while i < 100:
    valoresFunc1HC.append(alg.HillClimbing(f.func_1, -2, 0, True, 1.15, 300, 140))
    #valoresFunc1LRS.append(alg.lrs_search(-100, 100, 1.0, f.func_1))
    #valoresFunc1GRS.append(alg.grs_search(-100, 100,f.func_1))
    #valoresFunc1SAS.append(alg.simulated_annealing_search(-100, 100, f.func_1))
    i = i + 1
    print(i)


#TIRANDO A MODA
valoresFunc1HC_flat = [int(valor[0]) for valor, _ in valoresFunc1HC]  
moda_primeira_colunaHC = stats.mode(valoresFunc1HC_flat)
print("Moda das partes inteiras da primeira coluna de valoresFunc1HC:", moda_primeira_colunaHC)

valoresFunc1LRS_flat = [int(valor[0]) for valor, _ in valoresFunc1LRS]  
moda_primeira_colunaLRS = stats.mode(valoresFunc1LRS_flat)
print("Moda das partes inteiras da primeira coluna de valoresFunc1LRS:", moda_primeira_colunaLRS)

valoresFunc1GRS_flat = [int(valor[0]) for valor, _ in valoresFunc1GRS]  
moda_primeira_colunaGRS = stats.mode(valoresFunc1GRS_flat)
print("Moda das partes inteiras da primeira coluna de valoresFunc1GRS:", moda_primeira_colunaGRS)

valoresFunc1SAS_flat = [int(valor[0]) for valor, _ in valoresFunc1SAS]  
moda_primeira_colunaSAS = stats.mode(valoresFunc1SAS_flat)
print("Moda das partes inteiras da primeira coluna de valoresFunc1SAS:", moda_primeira_colunaSAS)


plt.show()