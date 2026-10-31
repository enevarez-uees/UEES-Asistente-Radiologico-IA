"""Clasificador proxy académico sintético, no diagnóstico."""
from functools import lru_cache
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# Etiquetas sintéticas de demostración, no pacientes ni validación clínica.
EJEMPLOS = [
('fractura del radio distal con desplazamiento',1),
('fractura del cúbito con edema de tejidos blandos',1),
('trazo de fractura en falange proximal',1),
('cambios degenerativos de columna lumbar',1),
('reducción del espacio intervertebral lumbar',1),
('fractura del quinto metacarpiano',1),
('fractura diafisaria de radio',1),
('signos de artrosis en articulaciones de la mano',1),
('desplazamiento óseo y luxación',1),
('alteración radiográfica compatible con lesión ósea',1),
('sin fracturas ni luxaciones visibles',0),
('alineación ósea conservada y espacios articulares normales',0),
('densidad ósea normal sin alteraciones evidentes',0),
('radiografía de mano derecha sin hallazgos agudos',0),
('radiografía de antebrazo sin alteraciones',0),
('columna lumbar con alineación conservada',0),
('tejidos blandos sin alteraciones evidentes',0),
('espacios articulares mantenidos',0),
('radio y cúbito de aspecto conservado',0),
('sin evidencia de lesión ósea aguda',0),
]

@lru_cache(maxsize=1)
def cargar_modelo():
    modelo = Pipeline([
        ('tfidf',TfidfVectorizer(strip_accents='unicode',ngram_range=(1,2))),
        ('mlp',MLPClassifier(hidden_layer_sizes=(16,),max_iter=450,random_state=42))
    ])
    modelo.fit([x[0] for x in EJEMPLOS],[x[1] for x in EJEMPLOS])
    return modelo

def clasificar(texto):
    modelo=cargar_modelo()
    pred=int(modelo.predict([texto])[0])
    confianza=float(max(modelo.predict_proba([texto])[0]))
    return pred, confianza
