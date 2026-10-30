# Problema, alcance y objetivos

## Problema
La redacción de informes radiológicos requiere transformar hallazgos expresados verbalmente en texto estructurado y consistente. El prototipo académico explora el uso de reconocimiento de voz y procesamiento de lenguaje natural para apoyar esta tarea, sin reemplazar al profesional responsable.

## Objetivo general
Desarrollar y evaluar un prototipo de apoyo a la elaboración de informes radiológicos que combine transcripción de voz, análisis de texto y recuperación de versos preconfigurados, con revisión humana antes de aprobar el resultado.

## Objetivos específicos
1. Permitir el dictado y la corrección manual del texto transcrito.
2. Evaluar un clasificador experimental TF-IDF + MLP para predecir si el informe contiene algún verso configurado.
3. Recuperar versos candidatos mediante similitud textual y filtrado por CPT.
4. Permitir la selección y edición del informe por una persona responsable.
5. Documentar métricas, errores, limitaciones y medidas de protección de datos.

## Alcance y exclusiones
El clasificador es binario y **no predice patologías**. La selección de versos es asistida, no automática ni validada clínicamente. La aplicación pública es una demostración con datos sintéticos; los resultados académicos de 272 registros pertenecen a otro conjunto de datos. No se autoriza su uso para decisiones diagnósticas.
