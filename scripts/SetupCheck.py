# scripts/01_setup_check.py

import tensorflow as tf
import os
import sys

print("✅ Verificação do ambiente\n")

# Python
print(f"Python: {sys.version.split()[0]}")

# TensorFlow e Keras
print(f"TensorFlow: {tf.__version__}")
print(f"Keras (embutido): {tf.keras.__version__}")

# GPU disponível
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detectada: {gpus[0].name}")
else:
    print("❌ Nenhuma GPU detectada.")

# Teste simples de operação
try:
    print("Testando multiplicação de matrizes...")
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print("✅ Multiplicação realizada com sucesso.")
except Exception as e:
    print(f"❌ Erro durante multiplicação: {e}")
