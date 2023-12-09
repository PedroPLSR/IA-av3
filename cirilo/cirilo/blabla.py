import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt



###FUNÇÕES###
class Funcs:
    
    def func_1(self, x, y):
        return x**2 + y**2

    def func_2(self,x, y):
        return np.exp(-(x**2 + y**2)) + 2 * np.exp(-((x - 1.7) ** 2 + (y - 1.7) ** 2))

    def func_3(self,x, y):
        return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + 20 + np.exp(1))

    def func_4(self,x, y):
        return((x**2 - (10*np.cos(2*np.pi*x)+ 10) + y**2 - (10*np.cos(2*np.pi*y)+ 10)))

    def func_5(self,x, y):
        return ((x - 1)**2 + 100 * (y - x**2)**2)

    def func_6(self,x, y):
        return ((x * np.sin(4*np.pi*x)) - (y*np.sin(4*np.pi*y + np.pi)) + 1)

    def func_7(self,x, y):
        return (np.sin(x) * (np.sin((x**2) / np.pi) ** (20)) - np.sin(y) * (np.sin((y**2) / np.pi) ** (20)))

    def func_8(self,x, y):
        return(-(y + 47) * (np.sin(np.sqrt(np.abs((x/2)+(y+47))))) - x * (np.sin(np.sqrt(np.abs((x) - (y + 47))))))
    
    
    # Gere dados para os eixos x e y
    
class Algorithms:
    def HillClimbing(self, func, initial_x=-1, initial_y=0, maximize=True, e=1.15, max_iterations=300, max_neighbors=140):
        Xbest = initial_x
        Ybest = initial_y
        Fbest = func(Xbest, Ybest)
        
        def generate_neighbor(value, epsilon):
            return np.random.uniform(low = value-epsilon, high = value + epsilon)
        
        is_better = lambda candidate, best: candidate > best if maximize else candidate < best
        
        i = 0
        while i < max_iterations:
            j = 0
            improvement = False
            
            while j < max_neighbors:
                j += 1
                x_candidate = generate_neighbor(Xbest, e)
                y_candidate = generate_neighbor(Ybest, e)
                candidate_fitness = func(x_candidate, y_candidate)
                
                if is_better(candidate_fitness, Fbest):
                    Xbest, Ybest = x_candidate, y_candidate
                    Fbest = candidate_fitness
                    improvement = True
                    break
            
            if not improvement:
                break
            
            i += 1
        
        return (Xbest, Ybest), Fbest


    def lrs_search(self, lower_bound, upper_bound, perturbation_sigma, func):
        MAXit = 800
        Xbest = np.random.normal(lower_bound, upper_bound)
        Ybest = np.random.normal(lower_bound, upper_bound)
        Fbest = func(Xbest, Ybest)

        i = 0
        while i < MAXit:
            
            nX = np.random.uniform(0, perturbation_sigma)
            nY= np.random.uniform(0, perturbation_sigma)
            
            xcand = Xbest + nX
            ycand = Ybest + nY
            
            if lower_bound <= xcand <= upper_bound and lower_bound <= ycand <= upper_bound :
                
                fcand = func(xcand, ycand)

                
                if fcand > Fbest:
                    Xbest = xcand
                    Ybest = ycand
                    Fbest = fcand
                    ax.scatter(Xbest,Ybest, Fbest, c="r", marker="x", s = 50)

            i += 1

        return (Xbest,Ybest), Fbest
    
    def grs_search(self, lower_bound, upper_bound, func):
        MAXit = 800
        Xbest = np.random.uniform(lower_bound, upper_bound)
        Ybest = np.random.uniform(lower_bound, upper_bound)
        Fbest = func(Xbest, Ybest)

        i = 0
        while i < MAXit:
            xcand = np.random.uniform(lower_bound, upper_bound)
            ycand = np.random.uniform(lower_bound, upper_bound)

            fcand = func(xcand, ycand)

            if fcand > Fbest:
                Xbest = xcand
                Ybest = ycand
                Fbest = fcand
                ax.scatter(Xbest,Ybest, Fbest, c="r", marker="*", s = 50)

            i += 1

        return (Xbest, Ybest), Fbest
    
    def simulated_annealing_search(self,lower_bound, upper_bound, func):
        Nmax = 800
        T = 1000.0  # Temperatura inicial
        alpha = 0.9  # Fator de redução da temperatura
        sigma = 7.5 # Parâmetro de perturbação aleatória

        Xbest = np.random.uniform(lower_bound, upper_bound)
        Ybest = np.random.uniform(lower_bound, upper_bound)
        Fbest = func(Xbest, Ybest)

        i = 0
        while i < Nmax:
            n = np.random.normal(0, sigma)
            xcand = Xbest + n
            ycand = Ybest + n

            # Verificar a violação da restrição em caixa
            if lower_bound <= xcand <= upper_bound and lower_bound <= ycand <= upper_bound:
                fcand = func(xcand, ycand)

                if fcand < Fbest or np.random.rand() < np.exp((Fbest - fcand) / T):
                    Xbest = xcand
                    Ybest = ycand
                    Fbest = fcand
                    ax.scatter(Xbest,Ybest, Fbest, c="purple", marker="*", s = 50)

            i += 1
            T *= alpha  # Reduz a temperatura

        return (Xbest, Ybest), Fbest



lowerBound = -2
upperBound = 5
x = np.linspace(lowerBound, upperBound, 100)
y = np.linspace(lowerBound, upperBound, 100)

# Crie uma grade de pontos (x, y)
X, Y = np.meshgrid(x, y)

f = Funcs()
# Calcule os valores da função para cada ponto na grade
Z = f.func_2(X, Y)

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

#USAR OS ALGORITMOS
#Comenta um, dps usa o outro etc, ficar mais clean o grafico
#alguns tem hiperparametro dentro, se nao conseguir fazer eles acharem a resposta eh so mudar um pouco os valores
alg = Algorithms()
alg.HillClimbing(f.func_2)
alg.lrs_search(lowerBound,upperBound,1,f.func_2)
alg.grs_search(lowerBound,upperBound,f.func_2)
alg.simulated_annealing_search(lowerBound,upperBound,f.func_2)
# Mostre o gráfico
#plt.show()