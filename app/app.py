"""UEES — demostración académica con audio, clasificación proxy y versos."""
from pathlib import Path
import re
import sys
import tempfile

import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buscar_versos import buscar_versos, diagnosticar_cpt
from src.modelo_demo import clasificar

st.set_page_config(page_title='UEES | Asistente Radiológico IA',page_icon='🩻',layout='centered')
st.title('🩻 Asistente Radiológico con IA')
st.caption('Proyecto Integrador Final · Maestría en Inteligencia Artificial · UEES')
st.warning('Prototipo académico con datos sintéticos. No realiza diagnósticos ni produce informes clínicos validados. Toda sugerencia exige revisión de un radiólogo.')

@st.cache_resource(show_spinner=False)
def whisper_model():
    import whisper
    return whisper.load_model('tiny')

st.subheader('1. Estudio y dictado')
cpt=st.selectbox('Código CPT',['72100','73090','73120'],format_func=lambda x:{'72100':'72100 · Columna lumbar','73090':'73090 · Antebrazo','73120':'73120 · Mano'}[x])
st.caption(f'Versos de demostración disponibles para este CPT: {diagnosticar_cpt(cpt)}')
audio=st.audio_input('Grabar dictado con el micrófono')
archivo=st.file_uploader('O cargar audio previamente grabado',type=['wav','mp3','m4a'])
fuente=audio if audio is not None else archivo
if 'texto' not in st.session_state: st.session_state.texto=''
if st.button('Transcribir audio con Whisper',disabled=fuente is None):
    sufijo=Path(getattr(fuente,'name','audio.wav')).suffix or '.wav'
    ruta=None
    try:
        with tempfile.NamedTemporaryFile(suffix=sufijo,delete=False) as f:
            f.write(fuente.getvalue())
            ruta=Path(f.name)
        with st.spinner('Transcribiendo el audio...'):
            resultado=whisper_model().transcribe(str(ruta),language='es',fp16=False)
        st.session_state.texto=resultado.get('text','').strip()
        st.success('Transcripción completada. Puedes corregirla antes de analizar.')
    except Exception as exc:
        st.error(f'No se pudo transcribir el audio ({type(exc).__name__}). Comprueba el formato y vuelve a intentarlo.')
    finally:
        if ruta is not None: ruta.unlink(missing_ok=True)
texto=st.text_area('Transcripción editable',key='texto',height=145,placeholder='El dictado aparecerá aquí. También puedes escribir un texto ficticio para probar.')

st.subheader('2. Clasificación experimental')
st.caption('TF-IDF + MLP sintético. La clase es una etiqueta proxy; no indica diagnóstico ni presencia real de una lesión.')
if st.button('Analizar y buscar versos',type='primary'):
    limpio=re.sub(r'\s+',' ',texto).strip()
    if len(limpio.split())<3:
        st.error('Ingresa al menos tres palabras para analizar.')
    else:
        pred,confianza=clasificar(limpio.lower())
        st.session_state.resultado={'texto':limpio,'cpt':cpt,'pred':pred,'confianza':confianza,'versos':buscar_versos(limpio,cpt)}
        st.session_state.seleccion=[]
        st.session_state.borrador=''
if 'resultado' in st.session_state:
    r=st.session_state.resultado
    if r['texto']!=re.sub(r'\s+',' ',texto).strip() or r['cpt']!=cpt:
        st.info('Cambiaste el texto o el CPT. Presiona «Analizar y buscar versos» para actualizar los resultados.')
    else:
        st.write(f"**Clase predicha:** `{r['pred']}` — "+('Texto asociado con presencia de verso (proxy)' if r['pred']==1 else 'Texto no asociado con presencia de verso (proxy)'))
        st.metric('Confianza estimada del clasificador',f"{r['confianza']:.1%}")
        st.caption('Esta confianza no es una probabilidad clínica calibrada. El CPT no es entrada del clasificador.')
        st.subheader('3. Sugerencia de versos radiológicos')
        st.caption('Búsqueda independiente por CPT y similitud TF-IDF/coseno. Una similitud baja no valida el contenido.')
        versos=r['versos']
        if not versos:
            st.warning('No existen versos configurados para este CPT en el catálogo de demostración.')
        else:
            opciones=list(range(len(versos)))
            elegidos=st.multiselect('Selecciona las frases que deseas incluir en el borrador',options=opciones,format_func=lambda i:f"{versos[i]['nombre']} · similitud {versos[i]['similitud']:.1%}")
            for i,v in enumerate(versos):
                with st.expander(f"{i+1}. {v['nombre']} — similitud {v['similitud']:.1%}"):
                    st.write(v['texto'])
                    st.caption(f"Código: {v['codigo']}")
            st.subheader('4. Borrador editable')
            if st.button('Crear borrador con las frases seleccionadas'):
                if not elegidos:
                    st.warning('Selecciona al menos una frase antes de crear el borrador.')
                else:
                    st.session_state.borrador='ESTUDIO CPT '+cpt+'\n\nHALLAZGOS:\n'+'\n'.join(versos[i]['texto'] for i in elegidos)+'\n\nCONCLUSIÓN:\n[PENDIENTE DE REVISIÓN PROFESIONAL]'
            st.text_area('Borrador para revisión profesional',key='borrador',height=220)
            st.caption('Las frases se incorporan solo si el usuario las selecciona. No se infieren hallazgos desde imágenes.')
            if st.session_state.borrador:
                st.download_button('Descargar borrador TXT',data=st.session_state.borrador,file_name='borrador_radiologico_demo.txt',mime='text/plain')
            st.checkbox('Confirmo que este borrador requiere validación de un profesional antes de cualquier uso',value=False)
