import numpy as np
import pandas as pd

df = pd.DataFrame({"A": [1, 2, np.nan], "B": [5, np.nan, np.nan], "C": [1, 2, 3]})
df

# soit on  laisse la missing data donc on laisserait tel quel mais cela ne marche clairement pas à tous les coups

# soit on supprime les rows qui contiennent des valeurs manquantes
df.dropna()  # supprime les lignes qui contiennent des valeurs manquantes
df.dropna(axis=1)  # supprime les colonnes qui contiennent des valeurs manquantes 
# comme d'habitude on peut précisesr inplace=True pour modifier le df d'origine si souhaité

# soit on remplace les valeurs manquantes par une valeur, par exemple 0
df.fillna(value="missing")  # remplace les valeurs manquantes par 0
df.fillna(value=df.mean())