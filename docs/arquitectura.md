# Arquitectura de la solución

## Flujo funcional
1. Entrada de audio para transcripción mediante Whisper.
2. Presentación del texto transcrito para corrección manual.
3. Evaluación experimental de texto mediante TF-IDF + MLP (objetivo binario: coincidencia con algún verso configurado).
4. Búsqueda de versos candidatos mediante TF-IDF y similitud coseno, filtrados por CPT.
5. Selección de verso, composición y edición del informe.
6. Revisión y aprobación humana.

## Componentes
- **Interfaz:** aplicación Streamlit.
- **ASR:** Whisper, utilizado para convertir voz en texto.
- **Clasificación experimental:** vectorización TF-IDF y red neuronal MLP.
- **Recuperación textual:** catálogo de versos y similitud coseno.
- **Persistencia del modelo de demostración:** artefacto `models/modelo_demo.joblib`.

## Separación de ambientes
La demostración pública emplea un conjunto sintético de 20 ejemplos de entrenamiento y cuatro versos de demostración. La evaluación académica utilizó 272 registros (201 entrenamiento y 71 evaluación final) y no debe atribuirse al modelo desplegado.

## Seguridad y limitaciones
No publicar audios, informes ni identificadores de pacientes. El texto transcrito puede contener errores; la revisión profesional es obligatoria. La arquitectura aquí descrita es funcional, no una certificación de seguridad o de aptitud clínica.
