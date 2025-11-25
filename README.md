# 🗳️ Visualizador Electoral México 2024

Dashboard interactivo para análisis geoespacial de datos electorales mexicanos.

## 🚀 Características

- ✅ Visualización por estado y nivel territorial (Sección, Distrito, Municipio)
- ✅ Mapas de calor con múltiples métricas
- ✅ Mapa de ganadores por partido
- ✅ Control de opacidad para ver etiquetas del mapa base
- ✅ Hover con información geográfica detallada
- ✅ Gráficos complementarios (partidos, participación)
- ✅ Descarga de imágenes en alta resolución

## 📦 Instalación Local

### Requisitos
- Python 3.11+
- Git

### Pasos

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/electoral-dashboard.git
cd electoral-dashboard

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar archivos de datos en carpeta data/
#    - maestro_electoral_con_metricascorregido.csv
#    - SECCION.shp (+ .shx, .dbf, .prj, .cpg)

# 5. Ejecutar aplicación
python app.py

# 6. Abrir en navegador
# http://localhost:8050