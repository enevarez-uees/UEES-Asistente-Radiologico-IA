\# UEES — Asistente Radiológico Experimental con Inteligencia Artificial



\## Proyecto Integrador Final — Maestría en Inteligencia Artificial



\*\*Título:\*\* Sistema de apoyo a la generación de informes radiológicos mediante reconocimiento de voz y procesamiento de lenguaje natural.



\## Descripción



Prototipo académico desarrollado para apoyar el proceso de elaboración de informes radiológicos mediante técnicas de Inteligencia Artificial.



La aplicación permite transcribir dictados mediante Whisper, corregir el texto obtenido, realizar una clasificación experimental mediante Machine Learning, buscar textos radiológicos predefinidos utilizando TF-IDF y similitud del coseno, y generar un borrador editable que requiere revisión profesional.



\## Aplicación pública



https://uees-asistente-radiologico-ia.streamlit.app/



La aplicación está desplegada en Streamlit Community Cloud y puede utilizarse desde un navegador web.



\## Funcionalidades



1\. \*\*Reconocimiento de voz:\*\* grabación y transcripción de dictados mediante OpenAI Whisper.

2\. \*\*Corrección manual:\*\* edición de la transcripción antes del análisis.

3\. \*\*Clasificación experimental:\*\* modelo TF-IDF + MLP que predice una etiqueta binaria proxy relacionada con la presencia de versos configurados.

4\. \*\*Búsqueda de versos:\*\* recuperación de textos relacionados mediante TF-IDF y similitud del coseno, considerando el CPT seleccionado.

5\. \*\*Generación de borradores:\*\* incorporación de versos seleccionados a un informe editable.

6\. \*\*Revisión y aprobación simulada:\*\* confirmación académica de revisión del contenido.



\## Modelos y datos



El proyecto contempla dos escenarios:



\- \*\*Modelo académico original:\*\* entrenado con 272 estudios históricos para tres códigos CPT. En la evaluación de la Semana 5 obtuvo F1 macro de 0.7434, recall de clase 1 de 0.5882 y exactitud de 0.8169.

\- \*\*Modelo público de demostración:\*\* entrenado con 20 ejemplos sintéticos. Se utiliza en Streamlit Cloud para permitir la demostración sin publicar información histórica potencialmente sensible.



El catálogo público contiene versos sintéticos de demostración. No representa el catálogo completo utilizado en el entorno local.



\*\*Las métricas del modelo académico original no corresponden al modelo sintético publicado.\*\*



\## Interpretación de los resultados



La clase 1 indica una asociación experimental con la presencia de algún verso, según una etiqueta proxy.



La clasificación no identifica enfermedades ni determina qué verso específico corresponde. La búsqueda de versos es un componente independiente.



Los porcentajes de confianza del clasificador no representan probabilidades clínicas calibradas. Los porcentajes de similitud miden semejanza textual, no certeza diagnóstica.



El CPT seleccionado no se utiliza como variable predictora del clasificador; sí permite filtrar la búsqueda de versos.



\## Ejecución local



1\. Instalar Python y las dependencias indicadas en `requirements.txt`.

2\. Instalar FFmpeg para el procesamiento de audio.

3\. Ejecutar la aplicación:



&#x20;  `python -m streamlit run app/app.py`



Para entrenar el modelo académico original, se requieren los archivos históricos correspondientes, utilizados únicamente de manera local y con autorización.



\## Privacidad y seguridad



\- No introducir nombres, cédulas ni información que permita identificar pacientes en la aplicación pública.

\- Los archivos históricos y el catálogo original no se publican en GitHub.

\- El modelo original `models/modelo\_inicial.joblib` permanece excluido del repositorio por precaución de privacidad.

\- Los artefactos `joblib` deben cargarse únicamente desde fuentes confiables.



\## Limitaciones



\- El reconocimiento de voz puede producir errores, especialmente con términos médicos.

\- La transcripción debe revisarse y corregirse antes de continuar.

\- El modelo académico original fue evaluado con informes escritos, no validado clínicamente con dictados reales.

\- El modelo público utiliza datos sintéticos y no cuenta con validación clínica.

\- Los versos sugeridos y los borradores requieren revisión humana.

\- La aprobación implementada es una simulación académica, no una firma médica.



\*\*Este sistema es exclusivamente un prototipo académico y no está destinado a emitir diagnósticos ni a utilizarse para decisiones médicas reales.\*\*

