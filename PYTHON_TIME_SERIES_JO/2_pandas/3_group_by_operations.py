# Un groupBy implique split, apply et combine → aavec pandas tout ça se faait via .groupby()

import pandas as pd

data = {
    "Company": ["GOOG", "GOOG", "MSFT", "MSFT", "FB", "FB"],
    "Person": ["Sam", "Charlie", "Amy", "Vanessa", "Carl", "Sarah"],
    "Sales": [200, 120, 340, 124, 243, 350],
}
df = pd.DataFrame(data)
df

df.groupby("Company")  # On peut grouper par une colonne, ici Company. Si on fait ça pandas fait le split mais il faut qu'on fasse l'aggreagate fonction sinon pas de return
df.groupby("Company")["Sales"].mean()
df.groupby("Company")["Sales"].max()
df.groupby("Company")["Sales"].count()
df.groupby("Company")["Sales"].describe()  # sympa de faire un describe pour avoir une vue d'ensemble
df.groupby("Company")["Sales"].describe().transpose()
