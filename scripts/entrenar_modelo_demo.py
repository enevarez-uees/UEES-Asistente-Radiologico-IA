
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]

# Ejemplos sintéticos, creados exclusivamente para la demostración.
# Clase 0: texto sin una descripción patológica configurada.
# Clase 1: texto con una descripción patológica de ejemplo.
ejemplos = [
    ("La alineación de la columna lumbar está conservada.", 0),
    ("Los cuerpos vertebrales mantienen su altura.", 0),
    ("No se observan alteraciones óseas evidentes.", 0),
    ("Los espacios intervertebrales están conservados.", 0),
    ("No se identifican trazos de fractura.", 0),
    ("Las estructuras óseas presentan aspecto conservado.", 0),
    ("La articulación mantiene su alineación habitual.", 0),
    ("No se observan cambios degenerativos.", 0),
    ("El radio y el cúbito mantienen su continuidad.", 0),
    ("Las articulaciones de la mano están conservadas.", 0),
    ("Se observan cambios degenerativos en la columna lumbar.", 1),
    ("Existe disminución del espacio intervertebral L4-L5.", 1),
    ("Se identifica reducción del espacio discal lumbar.", 1),
    ("Se observan osteofitos en los cuerpos vertebrales.", 1),
    ("Hay cambios degenerativos en las vértebras lumbares.", 1),
    ("Se observa un trazo de fractura en el radio distal.", 1),
    ("Se identifica fractura en el antebrazo.", 1),
    ("Existe alteración de la continuidad ósea del radio.", 1),
    ("Se observan cambios degenerativos interfalángicos.", 1),
    ("Hay signos de artrosis en las articulaciones de la mano.", 1),
]

textos = [texto for texto, _ in ejemplos]
etiquetas = [etiqueta for _, etiqueta in ejemplos]

modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2)
    )),
    ("modelo", MLPClassifier(
        hidden_layer_sizes=(16,),
        random_state=42,
        max_iter=1000
    ))
])

modelo.fit(textos, etiquetas)

destino = ROOT / "models" / "modelo_demo.joblib"
destino.parent.mkdir(exist_ok=True)
joblib.dump(modelo, destino)

print("Modelo sintético generado correctamente.")
print("Ejemplos de entrenamiento:", len(ejemplos))
print("Archivo:", destino)
