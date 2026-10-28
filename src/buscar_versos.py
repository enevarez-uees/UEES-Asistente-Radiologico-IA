
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
CATALOGO_LOCAL = ROOT / "data" / "catalogo_versos.csv"
CATALOGO_DEMO = ROOT / "data" / "catalogo_versos_demo.csv"
CATALOGO = CATALOGO_LOCAL if CATALOGO_LOCAL.exists() else CATALOGO_DEMO


def cargar_catalogo():
    df = pd.read_csv(CATALOGO, encoding="utf-8-sig")
    df["CODIGO_CPT"] = df["CODIGO_CPT"].astype(str).str.strip()
    df["NOMBRE_VERSO"] = df["NOMBRE_VERSO"].fillna("")
    df["TEXTO_VERSO"] = df["TEXTO_VERSO"].fillna("")
    return df


def buscar_versos(texto, cpt, limite=3):
    if not texto or not texto.strip():
        return []

    df = cargar_catalogo()
    df = df[df["CODIGO_CPT"] == str(cpt)].copy()

    if df.empty:
        return []

    df["contenido"] = (
        df["NOMBRE_VERSO"].astype(str) + " " +
        df["TEXTO_VERSO"].astype(str)
    )

    vectorizador = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2)
    )

    matriz = vectorizador.fit_transform(
        [texto] + df["contenido"].tolist()
    )

    similitudes = cosine_similarity(
        matriz[0:1], matriz[1:]
    ).flatten()

    df["similitud"] = similitudes

    # Umbral experimental de similitud textual
    UMBRAL_SIMILITUD = 0.15

    df = df[
        df["similitud"] >= UMBRAL_SIMILITUD
    ].copy()

    df = df.sort_values(
        "similitud", ascending=False
    ).head(limite)

    resultados = []

    for _, fila in df.iterrows():
        resultados.append({
            "codigo": fila["CODIGO_VERSO_PATOLOGICO"],
            "nombre": fila["NOMBRE_VERSO"],
            "texto": fila["TEXTO_VERSO"],
            "similitud": float(fila["similitud"])
        })

    return resultados
