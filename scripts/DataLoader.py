import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.utils import img_to_array

AUTOTUNE = tf.data.AUTOTUNE

class ImageLoader:
    def __init__(self, img_dir, labels_df, batch_size=16, target_size=(512, 512)):
        self.img_dir = img_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size

    def to_dataset(self, training: bool = True, shuffle_buffer: int = 4096, drop_remainder: bool = True, cache_in_ram: bool = False, cache_path: str | None = None):
        # 1) Pré-filtra caminhos existentes em Python (rápido e simples)
        pair_list = []
        for f, y in zip(self.labels_df["filename"].astype(str), self.labels_df["label"].astype(float)):
            p = os.path.normpath(os.path.join(self.img_dir, f))
            if os.path.exists(p):
                pair_list.append((p, float(y)))
        if not pair_list:
            raise RuntimeError("Nenhuma imagem válida encontrada após filtrar caminhos.")

        paths  = tf.constant([p for p, _ in pair_list], dtype=tf.string)
        labels = tf.constant([y for _, y in pair_list], dtype=tf.float32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

        def _parse_fn(path, label):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.target_size, antialias=True)
            img = tf.cast(img, tf.float32) / 255.0
            return img, tf.cast(label, tf.float32)

        ds = ds.map(_parse_fn, num_parallel_calls=AUTOTUNE, deterministic=False)

        # cache opcional
        if cache_path is not None:
            ds = ds.cache(cache_path)   # SSD/arquivo
        elif cache_in_ram:
            ds = ds.cache()             # RAM

        ds = ds.batch(self.batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(AUTOTUNE)
        return ds


"""    # ---------- Antigo: gerador numpy (mantido p/ compatibilidade) ----------
    def load_in_batches(self):
        total = len(self.labels_df)
        while True:
            for i in range(0, total, self.batch_size):
                batch_df = self.labels_df.iloc[i:i + self.batch_size]
                batch_images, batch_labels = [], []
                for _, row in batch_df.iterrows():
                    img_path = os.path.join(self.img_dir, str(row["filename"]))
                    if not os.path.exists(img_path):
                        print(f"⚠️ Imagem não encontrada: {img_path}")
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize(self.target_size)
                        img_array = img_to_array(img) / 255.0
                        batch_images.append(img_array)
                        batch_labels.append(float(row["label"]))
                    except Exception as e:
                        print(f"❌ Erro ao processar {img_path}: {e}")
                if batch_images:
                    yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)
"""                    
