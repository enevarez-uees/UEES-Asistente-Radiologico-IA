from src.buscar_versos import buscar_versos, diagnosticar_cpt
from src.modelo_demo import clasificar

def test_cpts():
    for cpt in ('72100','73090','73120'):
        assert diagnosticar_cpt(cpt)==3
        assert len(buscar_versos('alineación ósea conservada sin fracturas',cpt))==3

def test_clasificacion():
    p,c=clasificar('sin fracturas ni luxaciones visibles')
    assert p in (0,1) and 0<=c<=1
