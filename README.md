# UEES — Prototipo de clasificación radiológica

Esta primera versión reproduce **el modelo inicial** de la Semana 5 (TF-IDF + MLP), y ofrece una interfaz Streamlit. No realiza transcripción Whisper, selección de versos específicos ni generación de informes clínicos.

## Ejecución local

1. Instalar Python y dependencias: `pip install -r requirements.txt`
2. Colocar **localmente y de forma autorizada** `EjemploData(1).xlsx` y `VersosPatologicosDraMoncayo(3).xlsx` en la raíz de esta carpeta. Nunca publicarlos en GitHub.
3. Entrenar: `python src/train.py`
4. Iniciar interfaz: `streamlit run app/app.py`

## Qué significa la predicción

La clase 1 indica asociación con presencia de algún verso en el informe histórico según etiquetas proxy generadas automáticamente. No indica una enfermedad, ni identifica qué verso debe usarse. El CPT elegido no forma parte de las variables predictoras actuales.

## Limitaciones y seguridad

- Resultados del notebook Semana 5 (modelo inicial, test reservado): F1 macro 0.7434; recall clase 1 0.5882; exactitud 0.8169.
- Solo 272 estudios de tres CPT; no validado con audio, radiólogos ni pacientes nuevos.
- No introducir nombres, cédulas ni otros datos identificables en la interfaz.
- `models/modelo_inicial.joblib` es un artefacto entrenado con información potencialmente sensible y no debe publicarse sin revisión de privacidad y autorización.
- Los archivos `joblib` solo deben cargarse si provienen de una fuente de confianza.
- Este prototipo no está destinado a decisiones médicas.
