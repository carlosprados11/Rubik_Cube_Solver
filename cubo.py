# ----------------------------------------------------------------------
# Rubik's cube simulator
# Numpy is used for face representation and operation
# Matplotlib only for plotting
# Written by Miguel Hernando (2017)
# The aim of this code is to give a simple rubik cube simulator to
# test Discrete Planning Techniques.
# The code was developed for AI teaching purpose.
# Universidad Politécnica de Madrid

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import colors
from collections import deque

# Vector que contiene a todos los vértices y sus adyacencias
grafo = deque()


'''
Face State order as it is internally represented
    | 4 |
| 0 | 1 | 2 | 3 |
    | 5 |
Each face is represented by state matrix (NxN) and each cell is an integuer (0-5). 
Row and columns are disposed with the origin at the upper left corner, 
with faces disposed as the unfolded cube states. 

Rotations are referred to axis relative faces.
The outward-pointing normal of face 1 is the X axis.
The outward-pointing normal of face 2 is the Y axis.
The outward-pointing normal of face 4 is the Z axis.
 
Rotations are considered positive if they are ccw around the axis (math positive rotation)
The  cube slices are considered as layers. The upper layer (faces 1, 2 or 4) have index 0, while de 
backward layers (3,0,5) have index N-1 (N is the cube dimension)

Initial colors have the same index than their respective faces
'''

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class RubCube:
    # face + rotation, face -, lateral faces (index, [tuple 1] [tuple2) tomando como base la gira +
    # giro X
    F_axis = {'front': 1, 'back': 3, 'faces': ((2, (0, 1), (-1, 0)),
                                               (4, (-1, 0), (0, -1)),
                                               (0, (0, -1), (1, 0)),
                                               (5, (1, 0), (0,
                                                            1)))}  # giro F realizado en la cara 1  capa i afecta a la i*[0,i], (0...N)*[-i 0]
    # giro Y
    R_axis = {'front': 2, 'back': 0, 'faces': ((3, (0, 1), (-1, 0)),
                                               (4, (0, -1), (1, 0)),
                                               (1, (0, -1), (1, 0)),
                                               (5, (0, -1), (1,
                                                             0)))}  # giro R realizado en la cara 2  capa i afecta a la i*[0,i], (0...N)*[-i 0]
    # giro Z
    U_axis = {'front': 4, 'back': 5, 'faces': ((0, (1, 0), (0, 1)),
                                               (1, (1, 0), (0, 1)),
                                               (2, (1, 0), (0, 1)),
                                               (3, (1, 0), (0,
                                                            1)))}  # giro U realizado en la cara 4  capa i afecta a la i*[0,i], (0...N)*[-i 0]
    axis_dict = {'x': F_axis, 'y': R_axis, 'z': U_axis}

    def __init__(self, N=3):
        self._N = N
        self.reset()

    def rotate_90(self, axis_name='x', n=0, n_rot=1):
        '''rotates 90*n_rot around one axis ('x','y','z') the layer n'''
        if axis_name not in self.axis_dict:
            return
        axis = self.axis_dict[axis_name]
        if n == 0:  # rotate the front face
            self._state[axis['front']] = np.rot90(self._state[axis['front']], k=n_rot)
        if n == self._N - 1:
            self._state[axis['back']] = np.rot90(self._state[axis['back']], k=n_rot)
        aux = []
        for f in axis['faces']:
            if f[1][0] > 0:  # row +
                r = self._state[f[0]][n, ::f[2][1]]
            elif f[1][0] < 0:  # row -
                r = self._state[f[0]][-(n + 1), ::f[2][1]]
            elif f[1][1] > 0:  # column +
                r = self._state[f[0]][::f[2][0], n]
            else:
                r = self._state[f[0]][::f[2][0], -(n + 1)]
            aux.append(r)
        raux = np.roll(np.array(aux), (self._N) * n_rot)
        
        for i,f in enumerate(axis['faces']):
            r = raux[i]
            if f[1][0] > 0:  # row +
                self._state[f[0]][n, ::f[2][1]] = r
            elif f[1][0] < 0:  # row -
                self._state[f[0]][-(n + 1), ::f[2][1]] = r
            elif f[1][1] > 0:  # column +
                self._state[f[0]][::f[2][0], n] = r
            else:
                self._state[f[0]][::f[2][0], -(n + 1)] = r

    def set_State(self, state):
        self._state = np.array(state)

    def get_State(self):
        return totuple(self._state)

    def plot(self, block=True):
        plot_list = ((1, 4), (4, 0), (5, 1), (6, 2), (7, 3), (9, 5))
        color_map = colors.ListedColormap(['#00008f', '#cf0000', '#009f0f', '#ff6f00', 'w', '#ffcf00'], 6)
        fig = plt.figure(1, (8., 8.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(3, 4),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )
        for p in plot_list:
            grid[p[0]].matshow(self._state[p[1]], vmin=0, vmax=5, cmap=color_map)
        plt.show(block=block)

    def reset(self):
        self._state = []
        for i in range(6):
            self._state.append(i * np.ones((self._N, self._N), dtype=np.int8))
    def randomMoves(self, num):
        moves=[]
        for i in range(num):
            x = random.choice(('x','y','z'))
            num = random.randint(0, self._N - 1)
            n_rot = random.randint(-1,2)
            self.rotate_90(x,num,n_rot)
            moves.append((x,num,n_rot))
        return moves


# Clase objeto de cada vértice del grafo
class Vertice:
    iden = 0
    hijos = deque()
    padre = 0
    eje = ''
    fila = 0
    num = 0
    estado = 0
    
    # Inicialización de cada vértice
    def __init__(self, ide_n, padre_n, eje_n, fila_n, num_n, hijos_n):
        self.iden = ide_n        
        self.padre = padre_n        
        self.eje = eje_n
        self.fila = fila_n
        self.num = num_n
        self.hijos = hijos_n
       
    # Muestra información del vértice asociado
    def muestra_Datos(self):
        print('\nId: ',self.iden)
        print("Padre: ", self.padre)
        print('Eje: ', self.eje)
        print("Fila: ", self.fila)
        print("Num: ", self.num)


# Algoritmo de identificación del exito de una serie de acciones en el cubo
def Comprueba(cubo, nodo, resultado):
    
    acciones = deque()
    acciones.append(nodo)
    while grafo[nodo].padre != 0:
        nodo = grafo[nodo].padre
        acciones.append(nodo)
            
    while len(acciones)!=0:
        accion = acciones.pop()
        cubo.rotate_90(grafo[accion].eje, grafo[accion].fila, grafo[accion].num)
        
        # Comprobar si se ha resuelto
        if(resultado==cubo.get_State()):
            return 1
        
    acciones.clear()
    return 0


def define_Cubo(N):
    
    a = RubCube(N)
    
    # Movimientos realizados sobre el cubo
    a.rotate_90('z',1,2)
    a.rotate_90('y',1,1)
    a.rotate_90('z',2,1)
    #a.rotate_90('y',2,2)
    
    return a
    

# Algoritmo de búsqueda
def Busqueda(profundidad):
    
    ##### Algoritmo de búsqueda (DFS)
    for i in range(0, len(grafo)):
        grafo[i].estado = 0
        
    grafo[0].estado = 1;
    
    # Creamos cola de exploración
    cola = deque()
    for i in range(0, len(grafo[0].hijos)):
        cola.append(grafo[0].hijos[i])
        
    exito=0
    
    # Se sigue explorando
    while ((len(cola)!=0) and (exito==0)):
        
        # Sacamos un vertice de la cola
        if profundidad==1:
            u = cola.pop()
        else:
            u = cola.popleft()
        grafo[u].estado = 1
        
        a = define_Cubo(N)
        
        # Algoritmo de comprobación
        exito = Comprueba(a, u, resuelto)
        
        if exito==1:
            print("\nSOLUCION ENCONTRADA")
            acciones = deque()
            acciones.append(u)
            a = define_Cubo(N)
            
            while grafo[u].padre != 0:
                u = grafo[u].padre
                acciones.append(u)
            
            # Muestro el primer avance en la resolución
            a.plot()
            
            # Muestro el resto de avances
            while len(acciones)!=0:
                accion = acciones.pop()
                # Información sobre el camino tomado
                grafo[accion].muestra_Datos()
                a.rotate_90(grafo[accion].eje, grafo[accion].fila, grafo[accion].num)
                a.plot()
            acciones.clear()
            
        else:
            for i in range(0, len(grafo[u].hijos)):
                # Encolo los nuevos nodos
                cola.append(grafo[u].hijos[i])



if __name__ == '__main__':
    import sys

    try:
        N = int(sys.argv[1])
    except:
        N = 3

    # Cubo de partida
    a_ini = RubCube(N)
    
    # Cubo resuelto
    resuelto = a_ini.get_State()
    
    a = define_Cubo(N)
    
    # Variables para la generación del grafo
    profun = 4
    ide = 0
    collection_eje = ['x','y','z']
    collection_num = [-1, 1, 2]
    num_nodos = 0
    
    # Vertice inicial
    hijos = deque()
    x = Vertice(0,-1,'',-1,0,hijos)
    grafo.append(x)
    
    # Numero de nodos a crear
    b = 3*3*N
    for i in range(0, profun+1):
        num_nodos += b**i
    
    # Generación del grafo
    padre = 0
    
    while ide+1<num_nodos:
        for j in collection_eje:
            for k in range(0,3):
                for l in collection_num:
                    ide+=1  
                    hijos = deque()
                    x = Vertice(ide, padre, j, k, l, hijos)
                    grafo.append(x)
                     
        if ide==(b*padre + b):
            for m in range(ide-(b-1), ide+1):
                grafo[padre].hijos.append(m)
            padre = padre + 1
          
            
    print("\nGrafo generado")
    
    profundidad = 1
    Busqueda(profundidad)
    
    
    

        
