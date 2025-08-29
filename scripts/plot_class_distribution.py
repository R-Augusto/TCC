import pandas as pd
import matplotlib.pyplot as plt
import os

# Caminhos dos arquivos
RESULTS_DIR = "results"
train_path = os.path.join(RESULTS_DIR, "train_labels.csv")
val_path = os.path.join(RESULTS_DIR, "val_labels.csv")
test_path = os.path.join(RESULTS_DIR, "test_labels.csv")

# Carregar os arquivos CSV
df_train = pd.read_csv(train_path, names=["filename", "label"], header=None)
df_val = pd.read_csv(val_path, names=["filename", "label"], header=None)
df_test = pd.read_csv(test_path, names=["filename", "label"], header=None)

# Contagem por classe
train_counts = df_train["label"].value_counts().sort_index()
val_counts = df_val["label"].value_counts().sort_index()
test_counts = df_test["label"].value_counts().sort_index()

# Plotagem
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, counts, title in zip(
    axes,
    [train_counts, val_counts, test_counts],
    ["Treinamento", "Validação", "Teste"]
):
    counts.plot(kind="bar", ax=ax, color=["#4caf50", "#f44336"])
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Saudável (0)", "Doente (1)"])
    ax.set_ylabel("Quantidade de imagens")

plt.tight_layout()

# Salvar o gráfico
output_path = os.path.join("imagens", "distribuicao_classes_tcc.png")
plt.savefig(output_path)
print(f"✅ Gráfico salvo em: {output_path}")
