# UEES — Asistente Radiológico Experimental con IA

Prototipo académico para demostrar un flujo **dictado → Whisper → transcripción editable → clasificación proxy TF-IDF + MLP → recuperación por CPT con TF-IDF/coseno → borrador editable**. El borrador solo se construye con frases seleccionadas explícitamente por el usuario.

## Datos y alcance

La demostración contiene 20 ejemplos sintéticos para el clasificador y 9 versos sintéticos (3 por cada CPT 72100, 73090, 73120). Estos datos **no** corresponden a pacientes ni al modelo académico histórico. No se deben atribuir a esta versión las métricas del modelo académico de 272 estudios. La similitud textual no constituye una inferencia médica; no se analizan imágenes radiológicas.

## Ejecución

`pip install -r requirements.txt` y `streamlit run app/app.py`. Para audio, instalar también `ffmpeg` (Streamlit Cloud usa `packages.txt`). Whisper `tiny` se descarga la primera vez que se utiliza; se requiere conexión y tiempo para la descarga inicial.

## Despliegue

Publicar **el contenido completo de esta carpeta** en la raíz de un repositorio de Streamlit Community Cloud; archivo principal `app/app.py`. No subir datos históricos ni archivos médicos. Se incluye catálogo demo explícito para que los 3 CPT tengan resultados. La aplicación no accede a ningún repositorio ni cuenta por sí sola.

## Ejercicio demostrativo CPT 73090

«Radiografía de antebrazo derecho en dos proyecciones. No se observan fracturas ni luxaciones. La alineación del radio y el cúbito se encuentra conservada. La densidad ósea es normal. Los tejidos blandos no presentan alteraciones evidentes.»

Seleccionar 73090, grabar, transcribir, analizar, inspeccionar sugerencias, seleccionar frases apropiadas y crear borrador. Un profesional debe validar cualquier resultado.
