"""Reproduce el modelo inicial de Semana 5 sin publicar datos clínicos."""
from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

def normalizar(texto):
    texto = str(texto) if pd.notna(texto) else ''
    return re.sub(r'\s+', ' ', texto.replace('_x000D_', ' ')).strip().lower()

def main():
    historico = pd.read_excel(ROOT / 'EjemploData(1).xlsx')
    versos = pd.read_excel(ROOT / 'VersosPatologicosDraMoncayo(3).xlsx')
    cpts = ['73090', '73120', '72100']
    rx = historico[historico['CPT'].astype(str).str.strip().isin(cpts)].copy()
    vrx = versos[versos['CODIGO_CPT'].astype(str).str.strip().isin(cpts)].copy()
    rx['CPT'] = rx['CPT'].astype(str).str.strip()
    vrx['CODIGO_CPT'] = vrx['CODIGO_CPT'].astype(str).str.strip()
    rx['texto_completo'] = (rx['VALOR_RESULTADO'].fillna('').astype(str) + ' ' + rx['VALOR_RESULTADO_PART2'].fillna('').astype(str)).map(normalizar)
    rx = rx[rx['SECUENCIA_ORDEN'].notna()].copy()
    datos = rx.groupby(['SECUENCIA_ORDEN', 'CPT'], as_index=False).agg(informe=('texto_completo', ' '.join))
    versos_por_cpt = {}
    for _, fila in vrx.iterrows():
        frase = normalizar(fila['TEXTO_VERSO'])
        if frase:
            versos_por_cpt.setdefault(fila['CODIGO_CPT'], []).append(frase)
    def crear_caso(fila):
        frases = versos_por_cpt.get(fila['CPT'], [])
        texto = fila['informe']
        objetivo = int(any(f in texto for f in frases))
        for frase in frases:
            texto = texto.replace(frase, ' ')
        return pd.Series({'objetivo': objetivo, 'texto_sin_versos': ' '.join(texto.split())})
    datos = pd.concat([datos, datos.apply(crear_caso, axis=1)], axis=1)
    separador = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    itrain, itest = next(separador.split(datos, datos['objetivo'], groups=datos['SECUENCIA_ORDEN']))
    train, test = datos.iloc[itrain], datos.iloc[itest]
    modelo = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, min_df=2, ngram_range=(1, 2))),
        ('modelo', MLPClassifier(hidden_layer_sizes=(64, 32), alpha=.0001, max_iter=200, random_state=42, early_stopping=False)),
    ])
    modelo.fit(train['texto_sin_versos'].fillna('').astype(str), train['objetivo'].astype(int))
    pred = modelo.predict(test['texto_sin_versos'].fillna('').astype(str))
    y = test['objetivo'].astype(int)
    print('F1 macro:', round(f1_score(y, pred, average='macro'), 4))
    print('Recall clase 1:', round(recall_score(y, pred, zero_division=0), 4))
    print('Accuracy:', round(accuracy_score(y, pred), 4))
    carpeta = ROOT / 'models'
    carpeta.mkdir(exist_ok=True)
    joblib.dump(modelo, carpeta / 'modelo_inicial.joblib')
    print('Modelo guardado en models/modelo_inicial.joblib')

if __name__ == '__main__':
    main()
