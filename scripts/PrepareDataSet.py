
import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_PATH = "data/processed_labels.csv"
IMAGES_DIR = "data/images"
RESULTS_DIR = "results"
SEED = 42

def carregar_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {path}")
    df = pd.read_csv(path, header=None, names=["filename", "label"])
    return df

def filtrar_imagens_existentes(df):
    df["exists"] = df["filename"].apply(lambda x: os.path.exists(os.path.join(IMAGES_DIR, x)))
    df = df[df["exists"]]
    return df.drop(columns="exists")

def dividir_balanceado(df):
    df_0 = df[df["label"] == 0]
    df_1 = df[df["label"] == 1]

    def split(df_classe):
        train, temp = train_test_split(df_classe, test_size=0.3, random_state=SEED)
        val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
        return train, val, test

    train_0, val_0, test_0 = split(df_0)
    train_1, val_1, test_1 = split(df_1)

    df_train = pd.concat([train_0, train_1]).sample(frac=1, random_state=SEED)
    df_val = pd.concat([val_0, val_1]).sample(frac=1, random_state=SEED)
    df_test = pd.concat([test_0, test_1]).sample(frac=1, random_state=SEED)

    return df_train, df_val, df_test

def salvar_subconjuntos(df_train, df_val, df_test):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_train.to_csv(f"{RESULTS_DIR}/train_labels.csv", index=False)
    df_val.to_csv(f"{RESULTS_DIR}/val_labels.csv", index=False)
    df_test.to_csv(f"{RESULTS_DIR}/test_labels.csv", index=False)
    print("✅ Arquivos salvos em 'results/'.")

def mostrar_resumo(df, nome):
    print(f"\n📊 {nome.upper()}")
    print(df["label"].value_counts())
    print(f"Total: {len(df)} imagens")

if __name__ == "__main__":
    print("🔍 Lendo rótulos...")
    df = carregar_labels(LABELS_PATH)

    print("📁 Verificando imagens existentes...")
    df = filtrar_imagens_existentes(df)

    print("✂️ Dividindo em treino, validação e teste...")
    df_train, df_val, df_test = dividir_balanceado(df)

    mostrar_resumo(df_train, "Treinamento")
    mostrar_resumo(df_val, "Validação")
    mostrar_resumo(df_test, "Teste")

    salvar_subconjuntos(df_train, df_val, df_test)
