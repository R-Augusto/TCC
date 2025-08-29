import os
import pandas as pd

# Caminhos
csv_path = "data/processed_labels.csv"
images_dir = "data/images"

# Lê o CSV sem cabeçalho
df = pd.read_csv(csv_path, header=None)

# Coluna 0 = nome da imagem
image_filenames = df[0].tolist()

# Verificação das imagens existentes
missing = []
for filename in image_filenames:
    full_path = os.path.join(images_dir, filename)
    if not os.path.exists(full_path):
        missing.append(filename)

# Resultados
print(f"🔍 Total de imagens listadas: {len(image_filenames)}")
print(f"❌ Imagens ausentes: {len(missing)}")

if missing:
    print("➡️ Arquivos de imagem não encontrados:")
    for m in missing:
        print(f" - {m}")
else:
    print("✅ Todas as imagens estão presentes na pasta.")
