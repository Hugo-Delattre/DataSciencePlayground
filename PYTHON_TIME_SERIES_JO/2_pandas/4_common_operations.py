import pandas as pd
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [444, 544, 666, 444], 'col3': ['abc', 'def', 'ghi', 'xyz']})
df.head()

df['col2'].unique()  # Uniquement les valeurs uniques d'une colonne
df['col2'].nunique()  # Le nombre de valeurs uniques d'une colonne
df['col2'].value_counts()  # Le nombre de fois qu'une valeur unique apparaît dans une colonne, très utile

newdf = df[ (df['col1'] > 2) & (df['col2'] == 444)]  # On peut faire des filtres sur plusieurs colonnes]

def times_two(number):
    return number * 2

df['new_col'] = df['col1'].apply(times_two)  # On peut appliquer une fonction sur un dataframe
print(df)

del df['new_col']
# équivalent de df.drop('new_col', axis=1, inplace=True)  

df.columns
df.index
df.info()
df.shape
df.describe()

df.sort_values('col2')  # On peut trier un dataframe sur une colonne, ça sera par défaut ASC (croissant)
df.sort_values('col2', ascending=False)  # si on veut décroissant
df.isnull()  # On peut vérifier si il y a des valeurs nulles dans le dataframe
df.isnull().sum()  # On peut compter le nombre de valeurs nulles dans chaque colonne
df.dropna()  # On peut supprimer les lignes avec des valeurs nulles
df.dropna(how='all')  # On peut supprimer les lignes avec des valeurs nulles uniquement si toutes les colonnes sont nulles