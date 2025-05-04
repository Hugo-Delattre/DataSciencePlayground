# Une Series Pandas c'est built in sur les NumPy array :
# - list python = une liste
# - array numpy = une liste  + un index
# - Series pandas = une liste optimisée pour les calculs + un index + un index nommé

import numpy as np
import pandas as pd

from numpy.random import randn

labels = ["a", "b", "c"]
mylist = [10, 20, 30]
arr = np.array(mylist)
d = {"a": 10, "b": 20, "c": 30}
series_unamed = pd.Series(data=mylist)
series_named = pd.Series(data=mylist, index=labels)
print(series_unamed)  # l'index est par défaut 0, 1, 2
print(series_named)  # l'index est nommé (ici a, b, c)

ser = pd.Series([13, 32, 54], index=["France", "Japon", "Canada"])
print(ser)
ser.loc[
    "France"
]  # On peut alors accéder directement va l'index nommé plutôt que par l'index numérique
ser.iloc[0]  # On peut aussi accéder par l'index numérique

np.random.seed(101)
rand_mat = randn(5, 4)
rand_mat
df = pd.DataFrame(
    data=rand_mat
)  # Automatiquement formaté en colonnes à partir de notre matrice, mais il faut y ajouter le nom des colonnes et le nom des index
df2 = pd.DataFrame(
    data=rand_mat,
    index=["A", "B", "C", "D", "E"],
    columns=["Col1", "Col2", "Col3", "Col4"],
)
print(df2)
df2["Col1"]  # On peut accéder à une colonne par son nom
df2[
    ["Col1", "Col2"]
]  # On peut accéder à plusieurs colonnes en les mettant dans une liste
df2["Col1"]["A"]

df2["SommeCol1Col2"] = df2["Col1"] + df2["Col2"]
df2

df2.drop(
    columns="Col3", inplace=True
)  # On peut supprimer la colonne du df avec inplace=True
df2

df2.iloc[0]
df2.loc["A"]
# iloc pour index location, loc pour index nommé, ici les deux reviennent au même
df2.loc[['A', 'B']]
df2.loc[['A', 'B'], ["Col1", "Col2"]] # Comme pour sélectionner avec Numpy, 1 paramètre c'est les lignes, 2ème paramètre c'est les colonnes

df2 > 0
df2[df2 > 0] # On peut aussi faire un masque booléen pour sélectionner les valeurs

df2[df2["Col1"] > 0]  # On peut aussi faire un masque booléen sur une colonne, ne retourne que les lignes qui respectent la condition

df2[df2["Col1"] > 0]["Col2"]
df2[df2["Col1"] > 0]["Col2"].loc["A"]

cond1 = df2["Col1"] > 0
cond2 = df2["Col2"] > 1
# df2[cond1 and cond2] -> ça en pandas ça ne marche pas car la syntaxe python "and" est pour des booléens standard
# En Pandas il faut utiliser le & pour faire un "and" et le | pour faire un "or"
df2[(cond1) & (cond2)]

# .set_index() pour changer l'index nommé par une autre colonne

# df.info()

# df.dtypes pour voir les types de données

# df.describe() pour avoir un résumé statistique des données

df2.describe()

df2.head()  # Pour voir les premières lignes du df
df2.tail()  # Pour voir les dernières lignes du df

ser_col1 = df2["Col1"] > 0
ser_col1.value_counts()  # Pour voir le nombre de fois que chaque éléments du row est présent dans la série, 
sum(ser_col1)  # équivalent car true = 1 et false = 0 donc ça retourne le nombre de True
len(ser_col1)  # et ça pour voir le nombre total de valeurs