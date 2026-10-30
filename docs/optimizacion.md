# Optimización del modelo — Proyecto Integrador

## 1. Objetivo

Este documento presenta los experimentos de optimización y análisis de sensibilidad realizados sobre el clasificador TF-IDF + MLP del proyecto **Sistema de apoyo a la generación de informes radiológicos mediante reconocimiento de voz y procesamiento de lenguaje natural**.

El clasificador tiene como objetivo identificar si un informe contiene al menos uno de los versos configurados para el proyecto. **No identifica enfermedades ni realiza diagnósticos radiológicos.**

Se evaluó cómo diferentes configuraciones afectan el rendimiento del modelo y se analizaron sus limitaciones de generalización.

## 2. Metodología experimental

Se utilizaron 272 registros, distribuidos en:

- **201 registros de entrenamiento**, utilizados para los experimentos de optimización.
- **71 registros de evaluación final**, mantenidos separados durante la selección de hiperparámetros.

Los experimentos utilizaron validación cruzada agrupada mediante `GroupKFold`, con cinco particiones. El agrupamiento se realizó mediante `SECUENCIA_ORDEN`, para evitar compartir grupos entre entrenamiento y validación.

El procesamiento del texto se realizó mediante `TfidfVectorizer`, seguido de un clasificador `MLPClassifier`.

La métrica principal de comparación fue **F1 macro**, complementada con el recall de la clase positiva y la desviación estándar entre particiones.

Los experimentos de sensibilidad modificaron un hiperparámetro a la vez, manteniendo constantes los demás, salvo en el análisis específico de interacciones.

## 3. Sensibilidad a los hiperparámetros

### 3.1. Arquitectura de la red neuronal

| Arquitectura MLP | F1 macro promedio | Desviación estándar | Recall positivo |
|---|---:|---:|---:|
| (32,) | 0.8883 | 0.0582 | 0.7817 |
| (64, 32) | 0.9125 | 0.0386 | 0.8067 |
| (128, 64) | 0.9127 | 0.0533 | 0.8233 |

La arquitectura `(128, 64)` obtuvo el mayor F1 macro promedio, aunque la diferencia frente a `(64, 32)` fue de aproximadamente 0.0002.

Se seleccionó `(64, 32)` por ofrecer un rendimiento prácticamente equivalente, menor complejidad y menor variabilidad entre particiones.

### 3.2. Regularización (`alpha`)

| Alpha | F1 macro promedio | Recall positivo |
|---|---:|---:|
| 0.0001 | 0.9125 | 0.8067 |
| 0.001 | 0.9125 | 0.8067 |
| 0.01 | 0.9060 | 0.8067 |
| 0.1 | 0.8997 | 0.7900 |
| 1.0 | 0.8790 | 0.7567 |

Los resultados muestran una reducción del rendimiento con valores elevados de regularización. Se mantuvo `alpha=0.0001`, ya que proporcionó uno de los mejores resultados.

### 3.3. Tasa de aprendizaje (`learning_rate_init`)

| Learning rate | F1 macro promedio | Recall positivo |
|---|---:|---:|
| 0.0001 | 0.8419 | 0.6972 |
| 0.001 | 0.9125 | 0.8067 |
| 0.01 | 0.9099 | 0.8483 |

La tasa de aprendizaje de `0.001` obtuvo el mejor F1 macro promedio entre las configuraciones probadas.

El valor `0.01` produjo un recall positivo superior, pero un F1 macro ligeramente inferior.

Con `0.0001`, las cinco particiones alcanzaron el límite de 400 iteraciones sin converger. Por tanto, su menor rendimiento no debe atribuirse exclusivamente a la tasa de aprendizaje, sino también a un entrenamiento potencialmente insuficiente.

### 3.4. Características máximas TF-IDF (`max_features`)

| Max features | F1 macro promedio | Recall positivo |
|---|---:|---:|
| 500 | 0.8942 | 0.8067 |
| 1000 | 0.9125 | 0.8067 |
| 2000 | 0.9125 | 0.8067 |
| 3000 | 0.9125 | 0.8067 |

El rendimiento mejoró al aumentar el límite de 500 a 1000 características.

No se observaron mejoras al elevarlo a 2000 o 3000. El vectorizador entrenado con todos los registros de entrenamiento produjo 828 características, por debajo del límite de 1000.

Por este motivo, se mantuvo `max_features=1000`.

### 3.5. Número máximo de iteraciones

Se compararon los valores 200, 400 y 600 para `max_iter` con la arquitectura `(64, 32)` y la configuración base.

Los tres experimentos obtuvieron un F1 macro promedio aproximado de 0.9125. Con 400 iteraciones, las cinco particiones convergieron en esta configuración, sin mejoras adicionales al aumentar a 600.

Se seleccionó `max_iter=400`.

## 4. Comparación y clasificación descriptiva de sensibilidad

La amplitud observada del F1 macro entre los valores probados fue:

| Hiperparámetro | Diferencia entre F1 máximo y mínimo |
|---|---:|
| Tasa de aprendizaje | 0.0706 |
| Regularización | 0.0335 |
| Arquitectura MLP | 0.0244 |
| Características TF-IDF | 0.0183 |

La tasa de aprendizaje presentó la mayor variación observada. Sin embargo, este resultado está condicionado por la falta de convergencia del experimento con `learning_rate_init=0.0001`.

Este orden constituye una **clasificación descriptiva dentro de los rangos ensayados**, no una medida estadística universal de importancia ni una prueba de causalidad.

## 5. Análisis de interacción entre hiperparámetros

Se evaluaron dos arquitecturas y dos tasas de aprendizaje:

| Arquitectura | Learning rate | F1 macro promedio | Recall positivo |
|---|---:|---:|---:|
| (64, 32) | 0.001 | 0.9125 | 0.8067 |
| (64, 32) | 0.01 | 0.9099 | 0.8483 |
| (128, 64) | 0.001 | 0.9127 | 0.8233 |
| (128, 64) | 0.01 | 0.9043 | 0.8483 |

Los resultados sugieren que el cambio de tasa de aprendizaje puede afectar de manera diferente a cada arquitectura.

No obstante, las diferencias son pequeñas y no se realizaron pruebas de significancia estadística. Por ello, no puede afirmarse que exista una interacción estadísticamente significativa.

## 6. Configuración seleccionada

La configuración elegida fue:

| Componente | Parámetro | Valor |
|---|---|---|
| TF-IDF | `max_features` | 1000 |
| TF-IDF | `min_df` | 2 |
| TF-IDF | `ngram_range` | (1, 2) |
| MLP | `hidden_layer_sizes` | (64, 32) |
| MLP | `alpha` | 0.0001 |
| MLP | `learning_rate_init` | 0.001 |
| MLP | `max_iter` | 400 |
| MLP | `random_state` | 42 |

Esta selección busca un equilibrio entre rendimiento, estabilidad y complejidad.

El F1 macro promedio obtenido durante la validación cruzada agrupada fue aproximadamente **0.9125**.

## 7. Evaluación final independiente

Una vez seleccionada la configuración, se entrenó el modelo con los 201 registros de entrenamiento y se evaluó sobre los 71 registros separados.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8169 |
| Balanced accuracy | 0.7386 |
| F1 macro | 0.7434 |
| Recall de clase positiva | 0.5882 |
| Precisión de clase positiva | 0.6250 |
| F1 de clase positiva | 0.6061 |

Matriz de confusión:

| Clase real / Predicción | Predicción 0 | Predicción 1 |
|---|---:|---:|
| Real 0 | 48 | 6 |
| Real 1 | 7 | 10 |

Se identificaron siete falsos negativos entre 17 registros positivos, equivalentes a una tasa de falsos negativos de aproximadamente **41.18 %**.

Estos resultados muestran que el modelo todavía presenta limitaciones importantes para generalizar a registros no utilizados durante la optimización.

## 8. Análisis de sobreajuste y generalización

Se observaron los siguientes resultados:

- F1 macro sobre entrenamiento: **0.9722**.
- F1 macro promedio en validación cruzada: **0.9125**.
- F1 macro en evaluación final independiente: **0.7434**.

La diferencia entre entrenamiento y evaluación final fue de aproximadamente **0.2288**.

La diferencia entre validación cruzada y evaluación final fue de aproximadamente **0.1691**.

Estas diferencias son compatibles con problemas de generalización y posible sobreajuste. Sin embargo, no permiten establecer una causa única.

Se revisaron las distribuciones de clases y la longitud de los textos. No se identificó una diferencia evidente que, por sí sola, explicara la caída de rendimiento.

La optimización se realizó exclusivamente sobre los datos de entrenamiento; el conjunto de evaluación final no se utilizó para seleccionar hiperparámetros.

## 9. Análisis exploratorio de dependencia parcial

Se examinó cómo varía la probabilidad promedio estimada de la clase positiva al modificar individualmente dos características TF-IDF, manteniendo constantes las demás.

| Característica | Probabilidad inicial | Probabilidad final | Variación aproximada |
|---|---:|---:|---:|
| Fractura | 0.241772 | 0.239597 | −0.22 puntos porcentuales |
| Vertebrales | 0.241870 | 0.239967 | −0.19 puntos porcentuales |

Los cambios promedio fueron pequeños dentro de los rangos observados.

Estas curvas son exploratorias: modificar aisladamente una característica TF-IDF puede generar representaciones que no correspondan a documentos reales. Además, los promedios pueden ocultar efectos individuales e interacciones.

Por tanto, los resultados no constituyen explicaciones causales ni clínicas.

## 10. Limitaciones y consideraciones éticas

El proyecto presenta varias limitaciones:

1. El conjunto académico de datos es relativamente pequeño.
2. El clasificador tiene una tarea binaria específica y no detecta enfermedades.
3. El rendimiento de validación cruzada no se reprodujo en el conjunto de evaluación independiente.
4. Los falsos negativos observados requieren atención en futuras iteraciones.
5. Las curvas de dependencia parcial no representan relaciones clínicas causales.
6. El modelo académico y el modelo demostrativo desplegado públicamente corresponden a conjuntos de entrenamiento distintos.

La aplicación pública utiliza datos sintéticos de demostración. Las métricas del experimento académico **no deben atribuirse al modelo demostrativo**.

La solución se presenta como prototipo académico de apoyo a la elaboración de informes y requiere revisión humana. No está validada para uso clínico ni debe utilizarse para tomar decisiones diagnósticas.

## 11. Conclusiones

Los experimentos permitieron seleccionar una configuración razonable de TF-IDF + MLP y analizar cómo diferentes hiperparámetros afectan su rendimiento.

La configuración `(64, 32)` con `alpha=0.0001`, `learning_rate_init=0.001`, `max_iter=400` y `max_features=1000` mostró un buen equilibrio entre desempeño y complejidad durante la validación cruzada.

Sin embargo, la evaluación independiente obtuvo un F1 macro de 0.7434, considerablemente inferior al resultado promedio de validación cruzada. Esto evidencia que la optimización de hiperparámetros, por sí sola, no garantiza una mejor generalización.

El principal aprendizaje de esta etapa es la necesidad de evaluar los modelos con datos separados, interpretar críticamente sus métricas y reconocer sus limitaciones antes de considerar aplicaciones reales.

## 12. Evidencias y reproducibilidad

Los experimentos se desarrollaron en Google Colab y se conservaron en el notebook de optimización de la semana 5.

Se generaron archivos CSV con resultados de sensibilidad, interacciones y dependencia parcial; gráficos PNG; y un archivo JSON con las métricas de evaluación final.

Las evidencias agregadas pueden incorporarse al directorio `results/` del repositorio. Los datos radiológicos originales no deben publicarse sin la autorización correspondiente.

El código y los resultados deben mantenerse asociados a la versión del notebook con la que fueron generados para facilitar su revisión académica.
