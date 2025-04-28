import pandas as pd
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [444, 544, 666, 444], 'col3': ['abc', 'def', 'ghi', 'xyz']})
df.head()

df['col2'].unique()  # Uniquement les valeurs uniques d'une colonne
df['col2'].nunique()  # Le nombre de valeurs uniques d'une colonne
df['col2'].value_counts()  # Le nombre de fois qu'une valeur unique apparaît dans une colonne, très utile