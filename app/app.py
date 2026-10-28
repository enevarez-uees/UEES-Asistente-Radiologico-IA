"""Interfaz de demostración del clasificador experimental de versos."""
from pathlib import Path
import re
import os
import tempfile
import joblib
import streamlit as st
import whisper
from src.buscar_versos import buscar_versos


ROOT = Path(__file__).resolve().parents[1]

MODELO_ORIGINAL = ROOT / "models" / "modelo_inicial.joblib"
MODELO_DEMO = ROOT / "models" / "modelo_demo.joblib"

MODEL = MODELO_ORIGINAL if MODELO_ORIGINAL.exists() else MODELO_DEMO
ES_MODELO_DEMO = MODEL == MODELO_DEMO


st.set_page_config(page_title='UEES | Asistente radiológico experimental', page_icon='🩻', layout='centered')
st.title('🩻 Asistente radiológico experimental')
st.caption('Proyecto Integrador Final · Maestría en Inteligencia Artificial · UEES')
st.warning('Prototipo académico. No genera diagnósticos ni informes clínicos validados. Requiere revisión de un radiólogo.')

with st.sidebar:
    st.header('Acerca del prototipo')
    st.write('Clasifica si un texto radiológico está asociado con la presencia de algún verso configurado (etiqueta proxy).')
    
    if ES_MODELO_DEMO:
        st.info("Modelo activo: demostración sintética")
        st.write(
            "Entrenado con 20 ejemplos artificiales. "
            "No dispone de evaluación clínica ni de métricas "
            "comparables con el modelo académico original."
        )
    else:
        st.info("Modelo activo: académico original")
        st.write(
            "**Evaluación Semana 5 (modelo inicial):** "
            "F1 macro: 0.7434; recall clase 1: 0.5882; "
            "accuracy: 0.8169."
        )
        st.write(
            "Entrenado con informes históricos escritos, "
            "no con audios."
        )

    st.write(
        "La confianza del clasificador no representa "
        "una probabilidad clínicamente calibrada."
    )

    st.write('CPT admitidos en el experimento: 72100, 73090 y 73120.')

cpt = st.selectbox('Código CPT del estudio', ['72100', '73090', '73120'])

st.subheader('🎙️ Dictado radiológico')

audio = st.audio_input('Grabar dictado del radiólogo')

@st.cache_resource
def cargar_whisper():
    return whisper.load_model('small')

if audio is not None:
    st.audio(audio)

    if st.button('Transcribir audio con Whisper'):
        ruta_audio = None

        try:
            with st.spinner('Transcribiendo audio...'):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix='.wav'
                ) as archivo:
                    archivo.write(audio.getvalue())
                    ruta_audio = archivo.name

                modelo_whisper = cargar_whisper()
                resultado = modelo_whisper.transcribe(
                    ruta_audio,
                    language='es',
                    fp16=False
                )

                st.session_state['transcripcion'] = resultado['text'].strip()

            st.success('Transcripción completada.')

        except Exception as exc:
            st.error(
                f'Error durante la transcripción: {type(exc).__name__}: {exc}'
            )

        finally:
            if ruta_audio and os.path.exists(ruta_audio):
                os.remove(ruta_audio)

if 'transcripcion' in st.session_state:
    st.text_area(
        'Texto reconocido por Whisper (editable)',
        value=st.session_state['transcripcion'],
        height=130,
        key='texto_corregido'
    )
    st.info(
        'Revisa cuidadosamente la transcripción. '
        'Whisper puede cometer errores en términos médicos.'
    )

texto = st.text_area(
    'Texto radiológico para analizar (sin datos personales)',
    value=st.session_state.get('texto_corregido', ''),
    height=170,
    placeholder='Escribe aquí una descripción radiológica anonimizada...',
    key='texto_analisis'
)
if st.button('Prueba con ejemplo'):
    st.info('Ejemplo sintético: "se observa alineación ósea conservada sin alteraciones evidentes". Cópialo en el campo de texto para probar. No representa un diagnóstico.')

if st.button('Analizar texto', type='primary'):
    limpio = re.sub(r'\s+', ' ', texto).strip()
    if not limpio:
        st.error('Ingresa un texto antes de analizarlo.')
    elif len(limpio.split()) < 3:
        st.error('El texto es demasiado corto. Ingresa al menos tres palabras.')
    elif not MODEL.exists():
        st.error('No se encontró el modelo entrenado. Ejecuta primero: python src/train.py')
    else:
        try:
            with st.spinner('Procesando texto...'):
                modelo = joblib.load(MODEL)  # Solo cargar artefactos locales de confianza.
                pred = int(modelo.predict([limpio.lower()])[0])
                prob = modelo.predict_proba([limpio.lower()])[0]
            st.subheader('Resultado experimental')
            st.write('**Clase predicha:**', pred)
            st.write('**Interpretación:**', 'Texto asociado con presencia de verso (proxy)' if pred == 1 else 'Texto no asociado con presencia de verso (proxy)')
            st.metric('Confianza estimada del clasificador', f'{max(prob):.1%}')
            st.progress(float(max(prob)))
            st.info(f'CPT seleccionado: {cpt}. El modelo actual no usa CPT como variable de entrada; se muestra para contextualizar el estudio.')
            st.warning('No se identifica qué verso específico corresponde ni se genera un informe. Un profesional debe revisar cualquier uso posterior.')
        except Exception as exc:
            st.error(f'No fue posible procesar la entrada: {type(exc).__name__}. Revisa el modelo y las dependencias.')


st.divider()
st.subheader("📋 Sugerencia de versos radiológicos")

st.caption(
    "Búsqueda experimental mediante TF-IDF. "
    "Las sugerencias requieren validación del radiólogo."
)

if st.button("Buscar versos relacionados") or st.session_state.get("mostrar_versos", False):
    st.session_state["mostrar_versos"] = True
    texto_busqueda = st.session_state.get(
        "texto_analisis", ""
    ).strip()

    if not texto_busqueda:
        st.error("Primero ingresa o transcribe un texto.")
    else:
        try:
            resultados = buscar_versos(
                texto_busqueda, cpt
            )

            st.session_state["versos_encontrados"] = resultados

            if not resultados:
                st.warning(
                    "No existen versos configurados "
                    "para este CPT."
                )
            else:
                for verso in resultados:
                    with st.expander(
                        f"{verso['codigo']} - "
                        f"{verso['nombre']}"
                    ):
                        st.write(verso["texto"])
                        st.caption(
                            "Similitud textual: "
                            f"{verso['similitud']:.1%}"
                        )

                        st.checkbox(
                            "Seleccionar para el borrador",
                            key=f"seleccionar_verso_{verso['codigo']}"
                        )

        except Exception as exc:
            st.error(
                "No se pudo realizar la búsqueda: "
                f"{type(exc).__name__}: {exc}"
            )


st.divider()
st.subheader("📝 Borrador del informe radiológico")

if st.button("Generar borrador"):
    versos = st.session_state.get("versos_encontrados", [])

    seleccionados = [
        verso for verso in versos
        if st.session_state.get(
            f"seleccionar_verso_{verso['codigo']}", False
        )
    ]

    if not seleccionados:
        st.warning(
            "Selecciona al menos un verso antes de generar el borrador."
        )
    else:
        borrador = "\n\n".join(
            verso["texto"] for verso in seleccionados
        )

        st.session_state["borrador_informe"] = borrador
        st.success("Borrador generado correctamente.")

if "borrador_informe" in st.session_state:
    st.text_area(
        "Informe radiológico (editable)",
        key="borrador_informe",
        height=250
    )

    st.info(
        "Borrador experimental. El radiólogo debe revisar, "
        "corregir y aprobar todo el contenido antes de utilizarlo."
    )


st.divider()
st.subheader("✅ Revisión y aprobación del informe")

if "borrador_informe" in st.session_state:
    revisado = st.checkbox(
        "Confirmo que he revisado y corregido "
        "el contenido del informe radiológico."
    )

    if st.button("Aprobar informe"):
        if not revisado:
            st.warning(
                "Debes confirmar la revisión del informe "
                "antes de aprobarlo."
            )
        elif not st.session_state["borrador_informe"].strip():
            st.error("El informe no puede estar vacío.")
        else:
            st.session_state["informe_aprobado"] = (
                st.session_state["borrador_informe"]
            )
            st.success(
                "Informe marcado como aprobado "
                "en el prototipo académico."
            )
else:
    st.info(
        "Primero debes generar un borrador "
        "para habilitar la revisión."
    )
