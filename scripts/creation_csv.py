# CREATION CSV BEN
import os
import pandas as pd

# Chemin vers ton dossier contenant les images
dossier_images = r"D:\Desktop\dermproject\data\BEN"
fichier_csv = r"D:\Desktop\dermproject\data\ben.csv"

# Récupérer les noms des fichiers image
extensions_images = ('.jpg')
noms_images = [f for f in os.listdir(dossier_images) if f.lower().endswith(extensions_images)]

# Créer un DataFrame
df = pd.DataFrame(noms_images, columns=['image_id'])
df['target'] = 'benign'

# Sauvegarder en CSV
df.to_csv(fichier_csv, index=False)

print(f"Le fichier {fichier_csv} a été créé avec succès !")

# CREATION CSV MAL
import os
import pandas as pd

# Chemin vers ton dossier contenant les images
dossier_images = r"D:\Desktop\dermproject\data\MAL"
fichier_csv = r"D:\Desktop\dermproject\data\mal.csv"

# Récupérer les noms des fichiers image
extensions_images = ('.jpg')
noms_images = [f for f in os.listdir(dossier_images) if f.lower().endswith(extensions_images)]

# Créer un DataFrame
df = pd.DataFrame(noms_images, columns=['image_id'])
df['target'] = 'malignant'

# Sauvegarder en CSV
df.to_csv(fichier_csv, index=False)

print(f"Le fichier {fichier_csv} a été créé avec succès !")

# CONCATENATE 
import pandas as pd

ben = r"D:\Desktop\dermproject\data\ben.csv"
mal = r"D:\Desktop\dermproject\data\mal.csv"
csv_final = r"D:\Desktop\dermproject\data\dataset.csv"

# Charger les fichiers CSV en DataFrame
df_ben = pd.read_csv(ben)
df_mal = pd.read_csv(mal)
df_final = pd.concat([df_ben, df_mal], ignore_index=True)

df_final.to_csv(csv_final, index=False)