"""Pruebas unitarias con datos sintéticos, sin archivos clínicos."""
import pandas as pd
import pytest

from src import buscar_versos as modulo_versos
from src.train import normalizar


@pytest.fixture
def catalogo_sintetico(monkeypatch, tmp_path):
    archivo = tmp_path / "catalogo_pruebas.csv"
    pd.DataFrame([
        {"CODIGO_CPT": "73090", "CODIGO_VERSO_PATOLOGICO": "A",
         "NOMBRE_VERSO": "Fractura de mano",
         "TEXTO_VERSO": "Se observa fractura en los huesos de la mano"},
        {"CODIGO_CPT": "73090", "CODIGO_VERSO_PATOLOGICO": "B",
         "NOMBRE_VERSO": "Tejidos blandos",
         "TEXTO_VERSO": "Los tejidos blandos presentan aumento de volumen"},
        {"CODIGO_CPT": "72100", "CODIGO_VERSO_PATOLOGICO": "C",
         "NOMBRE_VERSO": "Columna vertebral",
         "TEXTO_VERSO": "Se observan cuerpos vertebrales alineados"},
    ]).to_csv(archivo, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(modulo_versos, "CATALOGO", archivo)
    return archivo


def test_normalizar_espacios_mayusculas_y_salto_excel():
    assert normalizar("  FRACTURA_x000D_  DE\n MANO  ") == "fractura de mano"


def test_busqueda_vacia_no_devuelve_resultados(catalogo_sintetico):
    assert modulo_versos.buscar_versos("  ", "73090") == []


def test_cpt_inexistente_no_devuelve_resultados(catalogo_sintetico):
    assert modulo_versos.buscar_versos("fractura de mano", "99999") == []


def test_busqueda_filtra_por_cpt_y_prioriza_coincidencia(catalogo_sintetico):
    resultados = modulo_versos.buscar_versos(
        "Se observa fractura en los huesos de la mano", "73090"
    )
    assert resultados
    assert resultados[0]["codigo"] == "A"
    assert all(item["codigo"] in {"A", "B"} for item in resultados)
    assert all(item["similitud"] >= 0.15 for item in resultados)


def test_limite_de_resultados_y_formato(catalogo_sintetico):
    resultados = modulo_versos.buscar_versos(
        "fractura mano tejidos blandos aumento de volumen", "73090", limite=1
    )
    assert len(resultados) <= 1
    if resultados:
        assert set(resultados[0]) == {"codigo", "nombre", "texto", "similitud"}
        assert isinstance(resultados[0]["similitud"], float)
