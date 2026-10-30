# Uso, demostración y limitaciones

## Demostración
Aplicación: https://uees-asistente-radiologico-ia.streamlit.app/

Flujo esperado: ingresar audio, revisar y corregir la transcripción, consultar sugerencias de versos filtradas por CPT, seleccionar una opción, editar el informe y confirmar la revisión humana. La disponibilidad de la aplicación debe verificarse al momento de la presentación.

## Casos de demostración sugeridos
- Dictado que coincide con un verso de ejemplo del catálogo.
- Dictado sin coincidencia clara que requiere edición y revisión humana.
- Corrección de una transcripción antes de incorporar texto al informe.

Estos son **guiones de demostración propuestos**, no evidencias de pruebas ya ejecutadas.

## Limitaciones
- Whisper puede cometer errores de reconocimiento.
- El clasificador experimental es binario y no diagnostica enfermedades.
- La búsqueda por similitud puede sugerir versos inadecuados.
- La muestra académica es pequeña y el desempeño reservado es inferior al de validación cruzada.
- La aplicación pública usa datos sintéticos y no reproduce las métricas académicas.
- No existe validación clínica ni autorización para uso asistencial.

## Revisión humana
El texto generado y las sugerencias deben ser revisados y corregidos por un profesional autorizado antes de cualquier uso externo.
