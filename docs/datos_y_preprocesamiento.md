# Datos y preprocesamiento

## Fuentes
El trabajo académico empleó registros históricos de informes radiológicos y un catálogo de versos configurados. Los archivos originales se mantienen fuera del repositorio público por contener información potencialmente sensible.

## Preparación
Los registros se agruparon por estudio y se construyó una etiqueta experimental binaria según coincidencia textual con los versos configurados. Se generó `texto_sin_versos` para reducir la filtración directa de la respuesta en el texto de entrada. Se utilizaron 272 registros: 201 para entrenamiento y validación cruzada, 71 reservados para evaluación final.

## División y control de filtración
Se utilizó división por grupos identificados mediante `SECUENCIA_ORDEN`. La optimización aplicó `GroupKFold(n_splits=5)` sobre los 201 registros de entrenamiento, ajustando TF-IDF dentro de cada partición mediante `Pipeline`.

## Vectorización
TF-IDF con `max_features=1000`, `min_df=2` y `ngram_range=(1,2)`. El vectorizador final aprendió 828 características.

## Riesgos de calidad
La etiqueta representa coincidencia textual, no diagnóstico. Los registros pueden contener diferencias de estilo, errores de transcripción o variaciones de terminología. Los archivos académicos privados no son necesarios para explorar la aplicación demostrativa, pero sí para reproducir íntegramente la evaluación histórica.
