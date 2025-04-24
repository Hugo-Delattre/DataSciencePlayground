import pandas as pd

print("test")

print("bonjour")

print(
    "Avec shift+enter on a bien un équivalent jupyter notebook depuis un fichier python (il faut pour cela avoir cocher l'option depuis les settings vscode)"
)

data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Location": ["New York", "Paris", "Berlin", "London"],
    "Age": [24, 13, 53, 33],
}

df = pd.DataFrame(data)
df
print(df)
