import numpy as np

arr = np.zeros(10)
print(arr)

arr2 = np.ones(10)
print(arr2)

arr3 = np.full(10, 5)
print(arr3)

arr4 = np.arange(10, 51)
print(arr4)

arr5 = np.arange(10, 51, 2)
print(arr5)

arr6 = np.arange(0,9).reshape(3, 3) # ✅ on crée un array 1D de 0 à 8, puis on le passe en 2D avec 3 lignes et 3 colonnes
print(arr6)

arr7 = np.eye(3)
print(arr7)

num8 = np.random.randint(0, 2) # ✅
print(num8)
num8_alt = np.random.rand(1)
print(num8_alt)

arr9 = np.random.randn(2)
print(arr9)
arr9_answer = np.random.randn(25)
print(arr9_answer)

arr10 = np.arange(1, 101).reshape(10,10)/100  # ✅
print(arr10)


arr10_alt = np.linspace(0.01, 1, 100).reshape(10,10)
print(arr10_alt)

arr11 = np.linspace(0, 1, 20)  # ✅ linspace génère un tableau de nombres également espacés sur un intervalle spécifié
print(arr11)

mat = np.arange(1,26).reshape(5,5)
mat

mat2 = mat[2:5, 1:5] #✅ (on pouvait aussi l'écrire mat[2:, 1:])
print(mat2)

mat3 = mat[-2, -1] # ✅ ou mat[3, 4] avec le premier étant la lligne, le deuxième la colonne
print(mat3)

mat4 = mat[0:3, 1:2] # casi ok mais le format attendu est 2D pas 1D, j'avais fait mat[0:3, 1], la soluce est de faire mat[0:3, 1:2] et là on a les mêmes résultats mais en 2D
print(mat4)

mat5 = mat[-1] #✅
print(mat5)

mat6 = mat[3:5] # ✅
print(mat6)

mat7 = np.sum(mat) # ✅ (on pouvait aussi faire mat.sum()
print(mat7)

mat8 = np.std(mat)
print(mat8)

mat9 = mat.sum(axis=0)
print(mat9)
