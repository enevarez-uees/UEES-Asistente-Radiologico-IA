"""Recuperación de frases radiológicas académicas, sin inferencia diagnóstica."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
CATALOGO_LOCAL = ROOT / 'data' / 'catalogo_versos.csv'
CATALOGO_DEMO = ROOT / 'data' / 'catalogo_versos_demo.csv'
CATALOGO = CATALOGO_LOCAL if CATALOGO_LOCAL.exists() else CATALOGO_DEMO
COLUMNAS = ['CODIGO_CPT','CODIGO_VERSO_PATOLOGICO','NOMBRE_VERSO','TEXTO_VERSO']

def cargar_catalogo():
    if not CATALOGO.exists():
        raise FileNotFoundError(f'Falta el catálogo: {CATALOGO}')
    df = pd.read_csv(CATALOGO, encoding='utf-8-sig', dtype={'CODIGO_CPT':str,'CODIGO_VERSO_PATOLOGICO':str})
    faltantes = set(COLUMNAS) - set(df.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas del catálogo: {sorted(faltantes)}')
    for col in COLUMNAS:
        df[col] = df[col].fillna('').astype(str).str.strip()
    df['CODIGO_CPT'] = df['CODIGO_CPT'].str.replace(r'\.0$', '', regex=True)
    return df

def diagnosticar_cpt(cpt):
    df = cargar_catalogo()
    return int((df['CODIGO_CPT'] == str(cpt).strip()).sum())

def buscar_versos(texto, cpt, limite=3):
    if not isinstance(texto, str) or not texto.strip():
        return []
    df = cargar_catalogo()
    df = df[(df['CODIGO_CPT'] == str(cpt).strip()) & (df['TEXTO_VERSO'] != '')].copy()
    if df.empty:
        return []
    df['contenido'] = df['NOMBRE_VERSO'] + ' ' + df['TEXTO_VERSO']
    matriz = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2)).fit_transform([texto] + df['contenido'].tolist())
    df['similitud'] = cosine_similarity(matriz[0:1], matriz[1:]).flatten()
    df = df.sort_values('similitud', ascending=False).head(max(1,int(limite)))
    return [{'codigo':r['CODIGO_VERSO_PATOLOGICO'],'nombre':r['NOMBRE_VERSO'],'texto':r['TEXTO_VERSO'],'similitud':float(r['similitud'])} for _,r in df.iterrows()]
