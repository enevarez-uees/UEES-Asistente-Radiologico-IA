# UEES — Asistente Radiológico Experimental con Inteligencia Artificial

**Proyecto Integrador Final — Maestría en Inteligencia Artificial, UEES**

**Título del proyecto:** Sistema de apoyo a la generación de informes radiológicos mediante reconocimiento de voz y procesamiento de lenguaje natural.

> **Aviso:** Prototipo exclusivamente académico. No es un dispositivo médico, no realiza diagnósticos y no debe utilizarse para decisiones clínicas ni con información identificable de pacientes.

## 1. Descripción y objetivo

Este proyecto explora un flujo asistido para elaborar borradores de informes radiológicos. Integra reconocimiento automático de voz, procesamiento de texto, clasificación experimental y recuperación de fragmentos predefinidos («versos»). El profesional conserva el control de la transcripción, la selección de textos y la revisión del informe.

**Problema abordado:** la elaboración manual de informes puede requerir transcripción, búsqueda de expresiones estandarizadas y edición repetitiva. El prototipo investiga si una combinación de herramientas de IA puede asistir estas tareas; **no se ha demostrado un beneficio clínico ni una reducción cuantificada de tiempos**.

## 2. Demostración pública

- **Aplicación Streamlit:** https://uees-asistente-radiologico-ia.streamlit.app/
- **Código fuente:** https://github.com/enevarez-uees/UEES-Asistente-Radiologico-IA

La demostración utiliza datos y versos sintéticos. No ingresar información personal o clínica real. La disponibilidad de la aplicación depende del servicio de alojamiento.

## 3. Flujo funcional

1. **Dictado y transcripción:** Whisper convierte audio en texto.
2. **Corrección humana:** el usuario revisa y modifica la transcripción.
3. **Clasificación experimental:** TF-IDF + red neuronal multicapa (MLP) predice una etiqueta binaria *proxy* relacionada con la presencia de algún verso configurado en el informe histórico.
4. **Búsqueda textual:** TF-IDF y similitud del coseno recuperan versos candidatos, filtrados por el código CPT elegido.
5. **Composición del borrador:** el usuario selecciona fragmentos y edita el informe.
6. **Revisión y aprobación simulada:** cierre académico del flujo, sin valor de firma médica.

**Importante:** el clasificador no diagnostica enfermedades ni elige un verso específico. La clasificación y la búsqueda son componentes distintos. El CPT filtra el catálogo de búsqueda, pero no es una variable predictora del clasificador.

## 4. Datos y modelos

| Entorno | Datos | Finalidad | Precaución |
|---|---|---|---|
| Académico original | 272 estudios históricos, tres códigos CPT | Entrenamiento, optimización y evaluación experimental | Datos originales no publicados |
| Demostración pública | 20 ejemplos sintéticos y catálogo reducido de versos sintéticos | Mostrar la interacción con la aplicación | Sus resultados no equivalen a los del modelo académico |

El modelo académico utiliza TF-IDF y MLP. La Semana 5 analizó arquitectura, regularización, tasa de aprendizaje y número de características de TF-IDF mediante validación cruzada agrupada. El conjunto independiente de prueba no se utilizó para seleccionar hiperparámetros.

## 5. Resultados experimentales

**Evaluación independiente del modelo académico original (71 estudios):**

| Métrica | Resultado |
|---|---:|
| Exactitud (*accuracy*) | 0.8169 |
| Exactitud balanceada | 0.7386 |
| F1 macro | 0.7434 |
| Recall de clase 1 | 0.5882 |
| Precisión de clase 1 | 0.6250 |
| F1 de clase 1 | 0.6061 |

Matriz de confusión: `[[48, 6], [7, 10]]`. En esta evaluación, el modelo no identificó 7 de los 17 casos positivos; esta limitación es relevante y **no permite afirmar idoneidad clínica**.

En validación cruzada agrupada, la configuración base seleccionada obtuvo un F1 macro medio aproximado de **0.9125**. La diferencia frente al F1 macro del conjunto de prueba (**0.7434**) evidencia una brecha de generalización que debe investigarse con más datos. Estos resultados **no corresponden al modelo público sintético**.

La confianza del clasificador no es una probabilidad clínica calibrada y la similitud del coseno no representa certeza diagnóstica.

**Evidencias y análisis:** [Optimización](docs/optimizacion.md) · [Resultados Semana 5](results/semana5/).

## 6. Organización del repositorio

```text
app/          Aplicación Streamlit
src/          Funciones de entrenamiento y búsqueda
scripts/      Entrenamiento del modelo sintético de demostración
data/         Catálogo público de demostración
models/       Modelo sintético de demostración
docs/         Documentación del proyecto
notebooks/    Notebooks académicos publicables, sin datos identificables
tests/        Pruebas automatizadas con datos sintéticos
results/      Métricas y gráficos de experimentación
```

Documentación complementaria disponible en [`docs/`](docs/). La existencia y el contenido de cada archivo deben verificarse en la versión publicada del repositorio.

## 7. Instalación y ejecución local

Requisitos: Python compatible con las dependencias del proyecto, `pip` y **FFmpeg** para procesamiento de audio.

```bash
git clone https://github.com/enevarez-uees/UEES-Asistente-Radiologico-IA.git
cd UEES-Asistente-Radiologico-IA
python -m venv .venv
```

Activar el entorno virtual:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Instalar dependencias y ejecutar:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Para ejecutar las pruebas unitarias:

```bash
python -m pip install pytest
python -m pytest -q tests/
```

El entrenamiento del modelo académico original requiere archivos históricos autorizados que **no están incluidos** en este repositorio. El modelo público se entrena por separado con datos sintéticos mediante `scripts/entrenar_modelo_demo.py`.

## 8. Seguridad, privacidad y ética

- No ingresar nombres, cédulas, imágenes ni audios identificables de pacientes en la aplicación pública.
- Los datos históricos y el catálogo original permanecen fuera del repositorio.
- El modelo académico original no se distribuye públicamente.
- Cargar archivos `joblib` únicamente desde fuentes confiables.
- Toda transcripción, sugerencia y composición debe ser revisada por una persona competente.
- La aprobación implementada es una simulación académica y no una firma profesional.

## 9. Limitaciones y trabajo futuro

- Posibles errores de Whisper con terminología médica, ruido y variaciones del dictado.
- Tamaño reducido de la muestra académica y brecha entre validación cruzada y prueba independiente.
- Etiqueta de clasificación *proxy*; no equivale a enfermedad, hallazgo radiológico validado ni pertinencia clínica.
- Recuperación basada en semejanza textual, no en comprensión diagnóstica.
- Modelo público con solo 20 ejemplos sintéticos y sin validación clínica.
- Falta de validación prospectiva con dictados reales, evaluación por especialistas y medición de impacto operativo.

Como continuidad se propone ampliar y depurar los datos autorizados, realizar evaluación prospectiva con especialistas, medir errores de transcripción y relevancia de versos, estudiar calibración e incorporar controles adicionales de privacidad y trazabilidad.

## 10. Material audiovisual de entrega

- Video Pitch: https://drive.google.com/file/d/1M7lSRGGsZjr-Xvchuf3VZXxtI-vOqnpb/view?usp=drive_link

- Video Respuesta de las Preguntas: https://drive.google.com/file/d/1BReqVvB8YvqfxpmLKxoM4PP9Sy4DQAG_/view?usp=drive_link

---

**Alcance final:** demostración académica de integración de IA y PLN con supervisión humana obligatoria; no apta para uso clínico real.
