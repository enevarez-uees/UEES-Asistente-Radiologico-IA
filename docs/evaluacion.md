# Evaluación del clasificador académico

## Diseño
La configuración se seleccionó con validación cruzada agrupada en 201 registros. El conjunto reservado de 71 registros se evaluó una vez tras seleccionar hiperparámetros.

## Configuración final
TF-IDF: `max_features=1000`, `min_df=2`, `ngram_range=(1,2)`. MLP: `hidden_layer_sizes=(64,32)`, `alpha=0.0001`, `learning_rate_init=0.001`, `max_iter=400`, `random_state=42`.

## Resultados
| Métrica | Valor |
|---|---:|
| F1 macro promedio CV agrupada | 0.9125 |
| F1 macro entrenamiento final | 0.9722 |
| Accuracy evaluación final | 0.8169 |
| Balanced accuracy evaluación final | 0.7386 |
| F1 macro evaluación final | 0.7434 |
| Recall clase positiva | 0.5882 |

Matriz de confusión (filas: real 0/1; columnas: predicción 0/1): `[[48,6],[7,10]]`. Los siete falsos negativos representan 41.18 % de los 17 positivos reservados.

## Interpretación
La caída de F1 macro desde CV (0.9125) hasta evaluación independiente (0.7434) señala limitaciones de generalización y es compatible con posible sobreajuste. No se afirma una mejora demostrada respecto de una línea base sobre el mismo conjunto reservado sin una comparación controlada adicional.

## Restricciones
Estas métricas **no** corresponden al modelo sintético de la aplicación pública. La etiqueta no corresponde a una patología ni a un resultado clínico.
