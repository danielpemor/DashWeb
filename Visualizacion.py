import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import numpy as np
import json
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# CONFIGURACIÓN DE COLORES Y DATOS
# ============================================================================
ESTADOS = {
    1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur',
    4: 'Campeche', 5: 'Coahuila', 6: 'Colima', 7: 'Chiapas', 8: 'Chihuahua',
    9: 'Ciudad de México', 10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero',
    13: 'Hidalgo', 14: 'Jalisco', 15: 'México', 16: 'Michoacán', 17: 'Morelos',
    18: 'Nayarit', 19: 'Nuevo León', 20: 'Oaxaca', 21: 'Puebla', 22: 'Querétaro',
    23: 'Quintana Roo', 24: 'San Luis Potosí', 25: 'Sinaloa', 26: 'Sonora',
    27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala', 30: 'Veracruz',
    31: 'Yucatán', 32: 'Zacatecas'
}

COORDS_ESTADOS = {
    1: {'lat': 21.88, 'lon': -102.28}, 2: {'lat': 30.54, 'lon': -115.72},
    3: {'lat': 26.04, 'lon': -111.67}, 4: {'lat': 19.83, 'lon': -90.53},
    5: {'lat': 27.06, 'lon': -101.71}, 6: {'lat': 19.24, 'lon': -103.72},
    7: {'lat': 16.75, 'lon': -93.12}, 8: {'lat': 28.63, 'lon': -106.08},
    9: {'lat': 19.43, 'lon': -99.13}, 10: {'lat': 24.56, 'lon': -104.66},
    11: {'lat': 21.02, 'lon': -101.26}, 12: {'lat': 17.44, 'lon': -99.54},
    13: {'lat': 20.09, 'lon': -98.76}, 14: {'lat': 20.67, 'lon': -103.35},
    15: {'lat': 19.29, 'lon': -99.65}, 16: {'lat': 19.57, 'lon': -101.71},
    17: {'lat': 18.68, 'lon': -99.10}, 18: {'lat': 21.75, 'lon': -104.85},
    19: {'lat': 25.59, 'lon': -99.99}, 20: {'lat': 17.07, 'lon': -96.72},
    21: {'lat': 19.04, 'lon': -98.20}, 22: {'lat': 20.59, 'lon': -100.39},
    23: {'lat': 19.18, 'lon': -88.48}, 24: {'lat': 22.15, 'lon': -100.98},
    25: {'lat': 25.00, 'lon': -107.48}, 26: {'lat': 29.30, 'lon': -110.33},
    27: {'lat': 17.98, 'lon': -92.93}, 28: {'lat': 24.27, 'lon': -98.84},
    29: {'lat': 19.32, 'lon': -98.24}, 30: {'lat': 19.53, 'lon': -96.93},
    31: {'lat': 20.71, 'lon': -89.09}, 32: {'lat': 22.77, 'lon': -102.58},
}

COLORES_PARTIDOS = {
    'PAN': '#0066CC', 'PRI': '#FF0000', 'PRD': '#FFD700',
    'PVEM': '#00A651', 'PT': '#C1272D', 'MC': '#FF6600',
    'MORENA': '#A0522D', 'PAN_PRI_PRD': '#9370DB',
    'PAN_PRI': '#8B7EC8', 'PAN_PRD': '#6B8DD6', 'PRI_PRD': '#D4A574',
    'PVEM_PT_MORENA': '#8B4513', 'PVEM_PT': '#C86428',
    'PVEM_MORENA': '#7D8B3F', 'PT_MORENA': '#9B5F4F',
    'COALICION_OPOSITORA': '#9370DB', 'COALICION_OFICIALISTA': '#8B4513',
}

COLORES_TIPO_SECCION = {
    'CRITICA_CONSOLIDAR': '#dc3545', 'DEFENSIVA_RIESGO': '#fd7e14',
    'OPORTUNIDAD_EXPANSION': '#28a745', 'MOVILIZABLE': '#17a2b8',
    'NORMAL': '#6c757d', 'CONSOLIDADA': '#007bff', 'BAJA_PRIORIDAD': '#e9ecef'
}

COLORES_TENDENCIA = {
    'CRECIMIENTO_SOSTENIDO': '#28a745', 'EXPANSION_RAPIDA': '#20c997',
    'CRECIMIENTO': '#7cb342', 'VOLATIL': '#ffc107', 'DECLIVE': '#ff9800',
    'DECLIVE_SOSTENIDO': '#fd7e14', 'DECLIVE_RAPIDO': '#dc3545',
    'AUGE_Y_CAIDA': '#e83e8c', 'RECUPERACION': '#17a2b8'
}

base_parties = ['PAN', 'PRI', 'PRD', 'PVEM', 'PT', 'MC', 'MORENA']
years = ['2012', '2018', '2024']
coaliciones = ['PAN_PRI_PRD', 'PAN_PRI', 'PAN_PRD', 'PRI_PRD',
               'PVEM_PT_MORENA', 'PVEM_PT', 'PVEM_MORENA', 'PT_MORENA']

# ============================================================================
# CLASE PRINCIPAL
# ============================================================================
class VisualizadorElectoral:
    def __init__(self, csv_path, shp_path):
        print("🔄 Cargando datos...")
        self.df = self.read_csv_file(csv_path)
        self.gdf = self.read_shapefile(shp_path)
        self.df_merged = self.merge_data()
        print(f"✅ Datos listos: {len(self.df_merged):,} registros fusionados\n")

    def read_csv_file(self, file_path):
        try:
            print("  📊 Cargando CSV...")
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            print(f"     {len(df):,} registros cargados")
            
            numeric_cols = [c for c in df.columns if c not in ['ID_ENTIDAD', 'SECCION']]
            numeric_cols = [c for c in numeric_cols if not c.startswith('TIPO_')]
            numeric_cols = [c for c in numeric_cols if not c.startswith('TENDENCIA_')]
            numeric_cols = [c for c in numeric_cols if c not in ['GANADOR_2024', 'SEGUNDO_2024', 'TIPO_SECCION']]
            
            for col in numeric_cols:
                if 'PARTICIPACION' in col or 'ABSTENCION' in col:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace('%', '').str.replace(',', '').replace('-', '0'), 
                        errors='coerce'
                    ).clip(0, 100)
                else:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace('%', '').str.replace(',', '').replace('-', '0'), 
                        errors='coerce'
                    )
            
            df['ID_ENTIDAD'] = pd.to_numeric(df['ID_ENTIDAD'], errors='coerce').astype('Int64')
            df['SECCION'] = pd.to_numeric(df['SECCION'], errors='coerce').astype('Int64')
            
            for year in years:
                total_col = f'TOTAL_VOTOS_{year}'
                if total_col not in df.columns:
                    base_cols = [f"{p}_{year}" for p in base_parties if f"{p}_{year}" in df.columns]
                    if base_cols:
                        df[total_col] = df[base_cols].sum(axis=1)
            
            df = self.calcular_coaliciones(df)
            print(f"     ✓ CSV procesado - {len(df.columns)} columnas")
            return df
            
        except Exception as e:
            print(f"❌ Error leyendo CSV: {str(e)}")
            raise

    def read_shapefile(self, shapefile_path):
        try:
            print("  🗺️  Cargando Shapefile...")
            gdf = gpd.read_file(shapefile_path, encoding='utf-8')
            print(f"     {len(gdf):,} geometrías cargadas")
            
            if 'ENTIDAD' in gdf.columns:
                gdf['ID_ENTIDAD'] = pd.to_numeric(gdf['ENTIDAD'], errors='coerce').astype('Int64')
            elif 'ID_ENTIDAD' in gdf.columns:
                gdf['ID_ENTIDAD'] = pd.to_numeric(gdf['ID_ENTIDAD'], errors='coerce').astype('Int64')
            
            gdf['SECCION'] = pd.to_numeric(gdf['SECCION'], errors='coerce').astype('Int64')
            
            if 'DISTRITO_F' in gdf.columns:
                gdf['DISTRITO_FEDERAL'] = pd.to_numeric(gdf['DISTRITO_F'], errors='coerce').astype('Int64')
            if 'DISTRITO_L' in gdf.columns:
                gdf['DISTRITO_LOCAL'] = pd.to_numeric(gdf['DISTRITO_L'], errors='coerce').astype('Int64')
            if 'MUNICIPIO' in gdf.columns:
                gdf['MUNICIPIO'] = pd.to_numeric(gdf['MUNICIPIO'], errors='coerce').astype('Int64')
            
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                print("     🔄 Reproyectando a WGS84...")
                gdf = gdf.to_crs(epsg=4326)
            
            print("     🔄 Simplificando geometrías...")
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001, preserve_topology=True)
            
            print(f"     ✓ Shapefile procesado")
            return gdf
            
        except Exception as e:
            print(f"❌ Error leyendo shapefile: {str(e)}")
            raise

    def calcular_coaliciones(self, df):
        def get_col_safe(df, col_name):
            return df[col_name].fillna(0) if col_name in df.columns else 0
        
        df['COALICION_OPOSITORA'] = (
            get_col_safe(df, 'PAN_2024') + get_col_safe(df, 'PRI_2024') + 
            get_col_safe(df, 'PRD_2024') + get_col_safe(df, 'PAN_PRI_PRD_2024')
        )
        
        df['COALICION_OFICIALISTA'] = (
            get_col_safe(df, 'MORENA_2024') + get_col_safe(df, 'PT_2024') + 
            get_col_safe(df, 'PVEM_2024') + get_col_safe(df, 'PVEM_PT_MORENA_2024')
        )
        
        df['MC_TOTAL'] = get_col_safe(df, 'MC_2024')
        
        def determinar_ganador_coalicion(row):
            votos = {
                'COALICION_OPOSITORA': row.get('COALICION_OPOSITORA', 0),
                'COALICION_OFICIALISTA': row.get('COALICION_OFICIALISTA', 0),
                'MC': row.get('MC_TOTAL', 0)
            }
            return max(votos, key=votos.get) if max(votos.values()) > 0 else 'SIN_DATOS'
        
        df['GANADOR_COALICION'] = df.apply(determinar_ganador_coalicion, axis=1)
        return df

    def merge_data(self):
        print("  🔗 Fusionando datos...")
        
        columnas_merge = []
        if 'SECCION' in self.gdf.columns and 'SECCION' in self.df.columns:
            columnas_merge.append('SECCION')
        if 'ID_ENTIDAD' in self.gdf.columns and 'ID_ENTIDAD' in self.df.columns:
            columnas_merge.append('ID_ENTIDAD')
        
        if not columnas_merge:
            raise ValueError("❌ No hay columnas comunes para hacer merge")
        
        print(f"     Fusionando por: {columnas_merge}")
        
        merged = self.gdf.merge(
            self.df, 
            on=columnas_merge, 
            how='left',
            suffixes=('_geo', '_data')
        )
        
        if len(columnas_merge) > 0:
            merged = merged.drop_duplicates(subset=columnas_merge, keep='first')
        
        print(f"     ✓ Merge completado: {len(merged):,} registros")
        return merged

    def agregar_por_nivel(self, nivel, estado_id=None):
        df = self.df_merged.copy()
        
        if estado_id:
            df = df[df['ID_ENTIDAD'] == estado_id]
        
        if nivel == 'SECCION':
            return df
        
        col_map = {
            'DISTRITO_FEDERAL': 'DISTRITO_FEDERAL',
            'DISTRITO_LOCAL': 'DISTRITO_LOCAL',
            'MUNICIPIO': 'MUNICIPIO'
        }
        
        col_agrupacion = col_map.get(nivel, nivel)
        
        if col_agrupacion not in df.columns:
            print(f"⚠️ Columna {col_agrupacion} no encontrada, usando SECCION")
            return df
        
        cols_sumar = []
        for year in years:
            for partido in base_parties:
                col = f"{partido}_{year}"
                if col in df.columns:
                    cols_sumar.append(col)
            total_col = f'TOTAL_VOTOS_{year}'
            if total_col in df.columns:
                cols_sumar.append(total_col)
        
        cols_sumar.extend(['COALICION_OPOSITORA', 'COALICION_OFICIALISTA', 'MC_TOTAL'])
        
        for coalicion in coaliciones:
            col = f'{coalicion}_2024'
            if col in df.columns:
                cols_sumar.append(col)
        
        for col in df.columns:
            if (('LISTA' in col or 'VOTOS' in col or 'NUM_CASILLAS' in col or 
                 'VOTOS_PARA_VOLTEAR' in col or 'VOTOS_GANADOS' in col or 'VOTOS_PERDIDOS' in col) and 
                col not in cols_sumar and not col.startswith('TIPO_')):
                if pd.api.types.is_numeric_dtype(df[col]):
                    cols_sumar.append(col)
        
        cols_sumar = [c for c in set(cols_sumar) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
        cols_promediar = []
        for col in df.columns:
            if ('PARTICIPACION' in col or 'ABSTENCION' in col or 
                'RETENCION' in col or 'SHARE' in col or 
                'PCT' in col or 'PORCENTAJE' in col or
                'COMPETITIVIDAD' in col or 'VOLATILIDAD' in col or
                'PRIORIDAD' in col or 'MARGEN' in col or
                'CRECIMIENTO_AJUSTADO' in col or 'CAMBIO_SHARE' in col or
                'NEP' in col or 'HHI' in col):
                if pd.api.types.is_numeric_dtype(df[col]) and not col.startswith('TIPO_'):
                    cols_promediar.append(col)
        
        agg_dict = {col: 'sum' for col in cols_sumar}
        agg_dict.update({col: 'mean' for col in cols_promediar})
        
        group_cols = [col_agrupacion]
        if 'ID_ENTIDAD' in df.columns and estado_id is None:
            group_cols.append('ID_ENTIDAD')
        
        gdf_temp = gpd.GeoDataFrame(df, geometry='geometry')
        
        try:
            # CORRECCIÓN: Buffer + Unary_union para cerrar gaps en niveles agregados
            if nivel in ['MUNICIPIO', 'DISTRITO_FEDERAL', 'DISTRITO_LOCAL']:
                print(f"  🔧 Procesando {nivel} (cerrando gaps)...")
                
                grouped = gdf_temp.groupby(group_cols)
                result_list = []
                
                for name, group in grouped:
                    # Validar y reparar geometrías antes de procesar
                    if not group.geometry.is_valid.all():
                        group = group.copy()
                        group['geometry'] = group.geometry.buffer(0)
                    
                    try:
                        # Buffer positivo pequeño para cerrar gaps
                        geoms_buffered = group.geometry.buffer(0.0008)
                        
                        # Unary union: fusiona todo en un solo polígono
                        merged_geom = geoms_buffered.unary_union
                        
                        # Buffer negativo para regresar al tamaño original
                        merged_geom = merged_geom.buffer(-0.0006)
                        
                        # Simplificar contorno
                        merged_geom = merged_geom.simplify(0.002, preserve_topology=True)
                        
                        # Validar geometría final
                        if not merged_geom.is_valid:
                            merged_geom = merged_geom.buffer(0)
                        
                    except Exception as geom_error:
                        print(f"    ⚠️ Error procesando geometría: {geom_error}")
                        merged_geom = group.geometry.unary_union
                        if not merged_geom.is_valid:
                            merged_geom = merged_geom.buffer(0)
                    
                    # Preparar datos agregados
                    row_data = {}
                    if len(group_cols) == 1:
                        row_data[group_cols[0]] = name
                    else:
                        for i, col in enumerate(group_cols):
                            row_data[col] = name[i]
                    
                    # Aplicar agregaciones
                    for col, func in agg_dict.items():
                        if col in group.columns:
                            try:
                                if func == 'sum':
                                    row_data[col] = float(group[col].sum())
                                elif func == 'mean':
                                    row_data[col] = float(group[col].mean())
                            except:
                                row_data[col] = 0.0
                    
                    row_data['geometry'] = merged_geom
                    result_list.append(row_data)
                
                gdf_dissolved = gpd.GeoDataFrame(result_list, crs=gdf_temp.crs)
                print(f"  ✅ {nivel}: {len(gdf_dissolved)} polígonos procesados")
            
            else:
                # Para SECCION, usar dissolve estándar
                gdf_dissolved = gdf_temp.dissolve(by=group_cols, aggfunc=agg_dict)
                gdf_dissolved = gdf_dissolved.reset_index()
        
        except Exception as e:
            print(f"⚠️ Error al disolver geometrías: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback seguro
            gdf_dissolved = gdf_temp.groupby(group_cols).agg(agg_dict).reset_index()
            geometry_col = gdf_temp.groupby(group_cols)['geometry'].first()
            gdf_dissolved = gdf_dissolved.merge(
                geometry_col.reset_index(), 
                on=group_cols, 
                how='left'
            )
            gdf_dissolved = gpd.GeoDataFrame(gdf_dissolved, geometry='geometry')
        
        # Recalcular coaliciones después de agregar
        gdf_dissolved = self.calcular_coaliciones(gdf_dissolved)
        
        # CORRECCIÓN: Convertir columnas Int64 a float para JSON (solo niveles agregados)
        if nivel in ['MUNICIPIO', 'DISTRITO_FEDERAL', 'DISTRITO_LOCAL']:
            for col in gdf_dissolved.columns:
                if str(gdf_dissolved[col].dtype) == 'Int64':
                    gdf_dissolved[col] = gdf_dissolved[col].fillna(0).astype('float64')
        
        return gdf_dissolved

    @staticmethod
    def _sanitize_for_json(gdf):
        """Convierte un GeoDataFrame a tipos JSON-safe (solo para niveles agregados)"""
        gdf_safe = gdf.copy()
        
        for col in gdf_safe.columns:
            if col == 'geometry':
                continue
            
            try:
                if pd.api.types.is_integer_dtype(gdf_safe[col]):
                    gdf_safe[col] = gdf_safe[col].apply(lambda x: int(x) if pd.notna(x) else 0)
                
                elif pd.api.types.is_float_dtype(gdf_safe[col]):
                    gdf_safe[col] = gdf_safe[col].apply(
                        lambda x: float(x) if pd.notna(x) and abs(x) != float('inf') else 0.0
                    )
                
                elif pd.api.types.is_bool_dtype(gdf_safe[col]):
                    gdf_safe[col] = gdf_safe[col].astype(bool)
                
                else:
                    gdf_safe[col] = gdf_safe[col].astype(str)
            
            except Exception as e:
                print(f"    ⚠️ Error convirtiendo columna {col}: {e}")
                gdf_safe[col] = gdf_safe[col].astype(str)
        
        return gdf_safe

    def crear_mapa(self, metrica, nivel='SECCION', estado_id=None, mostrar_ganador=False, opacidad=0.65):
        df_plot = self.agregar_por_nivel(nivel, estado_id)
        
        if len(df_plot) == 0:
            return go.Figure().add_annotation(
                text="No hay datos para mostrar",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color='red')
            )
        
        gdf_plot = gpd.GeoDataFrame(df_plot, geometry='geometry')
        
        if mostrar_ganador or metrica == 'Por partidos':
            return self._crear_mapa_ganador(gdf_plot, nivel, estado_id, opacidad)
        
        if metrica == 'TIPO_SECCION_ESTRATEGICA' and 'TIPO_SECCION_ESTRATEGICA' in gdf_plot.columns:
            return self._crear_mapa_tipo_seccion(gdf_plot, nivel, estado_id, opacidad)
        
        if 'TENDENCIA_HISTORICA' in metrica and metrica in gdf_plot.columns:
            return self._crear_mapa_tendencia(gdf_plot, metrica, nivel, estado_id, opacidad)
        
        if metrica not in df_plot.columns:
            return go.Figure().add_annotation(
                text=f"Métrica '{metrica}' no disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='orange')
            )
        
        # Preparar datos para plotly
        gdf_plot['id'] = gdf_plot.index
        
        es_participacion = 'PARTICIPACION' in metrica or 'ABSTENCION' in metrica or 'PCT' in metrica
        
        metricas_absolutas = [
            'LISTA_NOMINAL_2024', 'LISTA_NOMINAL_2018', 'LISTA_NOMINAL_2012',
            'TOTAL_VOTOS_2024', 'TOTAL_VOTOS_2018', 'TOTAL_VOTOS_2012',
            'VOTOS_PARA_VOLTEAR', 'MARGEN_VICTORIA_2024',
            'VOTOS_GANADOS_', 'VOTOS_PERDIDOS_',
            'VOLATILIDAD_HISTORICA_', 'VOLATILIDAD_TOTAL',
            'NEP_2024', 'HHI_2024',
            'PRIORIDAD_MOVILIZACION', 'COMPETITIVIDAD',
            'CANDIDATO_NO_REGISTRADO_2024', 'VOTOS_NULOS_2024'
        ]
        
        es_metrica_absoluta = any(abs_metric in metrica for abs_metric in metricas_absolutas)
        
        if es_participacion or 'SHARE' in metrica or 'RETENCION' in metrica or 'CRECIMIENTO_AJUSTADO' in metrica or 'CAMBIO_SHARE' in metrica:
            column_to_plot = metrica
            gdf_plot[column_to_plot] = gdf_plot[column_to_plot].clip(0, 100)
            max_val = 100
            color_scale = 'Reds' if 'PARTICIPACION' in metrica else 'Oranges'
        elif es_metrica_absoluta:
            column_to_plot = metrica
            gdf_plot[column_to_plot] = gdf_plot[column_to_plot].fillna(0)
            gdf_plot.loc[gdf_plot[column_to_plot] == 0, column_to_plot] = float('nan')
            max_val = gdf_plot[column_to_plot].quantile(0.95)
            color_scale = 'Blues' if 'LISTA_NOMINAL' in metrica or 'TOTAL_VOTOS' in metrica else 'Reds'
        else:
            year = next((y for y in years if metrica.endswith(f'_{y}')), '2024')
            total_column = f'TOTAL_VOTOS_{year}'
            
            if total_column not in gdf_plot.columns:
                column_to_plot = metrica
                gdf_plot[column_to_plot] = gdf_plot[column_to_plot].fillna(0)
                max_val = gdf_plot[column_to_plot].quantile(0.95)
                color_scale = 'Reds'
            else:
                gdf_plot['porcentaje'] = gdf_plot.apply(
                    lambda x: (x[metrica] / x[total_column] * 100) if x[total_column] > 0 else 0,
                    axis=1
                )
                column_to_plot = 'porcentaje'
                gdf_plot.loc[gdf_plot[column_to_plot] == 0, column_to_plot] = float('nan')
                max_val = gdf_plot[column_to_plot].quantile(0.95)
                color_scale = 'Reds'
        
        # Validar geometrías antes de convertir
        if not gdf_plot.geometry.is_valid.all():
            print("  🔧 Reparando geometrías inválidas...")
            gdf_plot['geometry'] = gdf_plot.geometry.buffer(0)
        
        # CORRECCIÓN: Sanitizar SOLO si NO es nivel SECCION
        if nivel in ['MUNICIPIO', 'DISTRITO_FEDERAL', 'DISTRITO_LOCAL']:
            print(f"  🔧 Convirtiendo tipos de datos para JSON ({nivel})...")
            gdf_plot_safe = self._sanitize_for_json(gdf_plot)
            geojson = json.loads(gdf_plot_safe.to_json())
        else:
            # Para SECCION, conversión normal
            geojson = json.loads(gdf_plot.to_json())
        
        center_coords = COORDS_ESTADOS.get(estado_id, {'lat': 23.6345, 'lon': -102.5528})
        zoom_level = 7 if estado_id else 4
        
        fig = px.choropleth_mapbox(
            gdf_plot,
            geojson=geojson,
            locations='id',
            color=column_to_plot,
            hover_name='SECCION' if nivel == 'SECCION' else nivel,
            hover_data={
                metrica: ':.2f',
                column_to_plot: ':.2f',
                'id': False
            },
            mapbox_style='carto-positron',
            zoom=zoom_level,
            center=center_coords,
            opacity=opacidad,
            color_continuous_scale=color_scale,
            labels={column_to_plot: metrica},
            range_color=[0, max_val]
        )
        
        estado_nombre = ESTADOS.get(estado_id, 'Nacional') if estado_id else 'Nacional'
        
        fig.update_layout(
            title={
                'text': f'<b>{metrica}</b><br><sub>{estado_nombre} - {nivel}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            margin={'r': 0, 't': 60, 'l': 0, 'b': 0},
            paper_bgcolor='#f8f9fa',
            font={'family': 'Arial, sans-serif'},
            coloraxis_colorbar=dict(
                title='',
                ticksuffix='%' if es_participacion or 'SHARE' in metrica or 'RETENCION' in metrica or 'CRECIMIENTO' in metrica else '',
                thickness=15,
                len=0.7,
                x=1.02
            ),
            height=700
        )
        
        # CORRECCIÓN: Bordes según nivel
        if nivel == 'SECCION':
            fig.update_traces(
                marker_line_width=0.2, 
                marker_line_color='rgba(50, 50, 50, 0.3)'
            )
        else:
            # Niveles agregados: SIN BORDES
            fig.update_traces(
                marker_line_width=0,
                marker_line_color='rgba(0, 0, 0, 0)'
            )
        
        return fig

    def _crear_mapa_ganador(self, gdf_plot, nivel, estado_id, opacidad=0.65):
        party_cols_2024 = [f"{p}_2024" for p in base_parties if f"{p}_2024" in gdf_plot.columns]
        
        if not party_cols_2024:
            return go.Figure().add_annotation(
                text="No hay datos de partidos para 2024",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color='orange')
            )
        
        gdf_plot['PARTIDO_PREDOMINANTE'] = gdf_plot[party_cols_2024].idxmax(axis=1).str.replace('_2024', '')
        gdf_plot['VOTOS_GANADOR'] = gdf_plot[party_cols_2024].max(axis=1)
        gdf_plot['TOTAL_VOTOS'] = gdf_plot[party_cols_2024].sum(axis=1)
        gdf_plot['PORCENTAJE_GANADOR'] = (gdf_plot['VOTOS_GANADOR'] / gdf_plot['TOTAL_VOTOS'] * 100).fillna(0)
        gdf_plot['COLOR'] = gdf_plot['PARTIDO_PREDOMINANTE'].map(COLORES_PARTIDOS)
        gdf_plot['id'] = gdf_plot.index
        
        # Validar geometrías
        if not gdf_plot.geometry.is_valid.all():
            gdf_plot['geometry'] = gdf_plot.geometry.buffer(0)
        
        # CORRECCIÓN: Sanitizar solo niveles agregados
        if nivel in ['MUNICIPIO', 'DISTRITO_FEDERAL', 'DISTRITO_LOCAL']:
            gdf_plot = self._sanitize_for_json(gdf_plot)
        
        center_coords = COORDS_ESTADOS.get(estado_id, {'lat': 23.6345, 'lon': -102.5528})
        zoom_level = 7 if estado_id else 4
        
        fig = go.Figure()
        
        partidos_presentes = gdf_plot['PARTIDO_PREDOMINANTE'].unique()
        
        for partido in partidos_presentes:
            df_partido = gdf_plot[gdf_plot['PARTIDO_PREDOMINANTE'] == partido]
            
            if len(df_partido) == 0:
                continue
            
            color = COLORES_PARTIDOS.get(partido, '#888888')
            
            hover_text = []
            for idx, row in df_partido.iterrows():
                estado_nombre = ESTADOS.get(row.get('ID_ENTIDAD'), 'N/A')
                
                text = f"<b>{partido}</b><br>Estado: {estado_nombre}<br>"
                
                if nivel == 'SECCION':
                    seccion = row.get('SECCION', 'N/A')
                    text += f"Sección: {seccion}<br>"
                
                text += (
                    f"<br>"
                    f"Votos: {row['VOTOS_GANADOR']:,.0f}<br>"
                    f"Porcentaje: {row['PORCENTAJE_GANADOR']:.1f}%<br>"
                    f"Total: {row['TOTAL_VOTOS']:,.0f}"
                )
                hover_text.append(text)
            
            fig.add_trace(go.Choroplethmapbox(
                geojson=json.loads(df_partido.to_json()),
                locations=df_partido['id'],
                z=[1] * len(df_partido),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                marker_line_width=0.2 if nivel == 'SECCION' else 0,
                marker_line_color='rgba(50,50,50,0.3)' if nivel == 'SECCION' else 'rgba(0,0,0,0)',
                marker_opacity=opacidad,
                hovertext=hover_text,
                hoverinfo='text',
                name=partido
            ))
        
        estado_nombre = ESTADOS.get(estado_id, 'Nacional') if estado_id else 'Nacional'
        
        fig.update_layout(
            mapbox_style='carto-positron',
            mapbox_zoom=zoom_level,
            mapbox_center=center_coords,
            title={
                'text': f'<b>Mapa de Predominancia Electoral 2024</b><br><sub>{estado_nombre} - {nivel}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            margin={'r': 0, 't': 60, 'l': 0, 'b': 0},
            paper_bgcolor='#f8f9fa',
            font={'family': 'Arial, sans-serif'},
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#333333",
                borderwidth=1
            ),
            height=700
        )
        
        return fig

    def _crear_mapa_tipo_seccion(self, gdf_plot, nivel, estado_id, opacidad=0.65):
        center_coords = COORDS_ESTADOS.get(estado_id, {'lat': 23.6345, 'lon': -102.5528})
        zoom_level = 7 if estado_id else 4
        
        gdf_plot['id'] = gdf_plot.index
        
        # CORRECCIÓN: Sanitizar solo niveles agregados
        if nivel in ['MUNICIPIO', 'DISTRITO_FEDERAL', 'DISTRITO_LOCAL']:
            gdf_plot = self._sanitize_for_json(gdf_plot)
        
        fig = go.Figure()
        
        orden_tipos = [
            'CRITICA_CONSOLIDAR', 'DEFENSIVA_RIESGO', 'OPORTUNIDAD_EXPANSION',
            'MOVILIZABLE', 'NORMAL', 'CONSOLIDADA', 'BAJA_PRIORIDAD'
        ]
        
        tipos_presentes = gdf_plot['TIPO_SECCION_ESTRATEGICA'].dropna().unique()
        tipos_presentes = [t for t in orden_tipos if t in tipos_presentes]
        
        for tipo in tipos_presentes:
            df_tipo = gdf_plot[gdf_plot['TIPO_SECCION_ESTRATEGICA'] == tipo]
            
            if len(df_tipo) == 0:
                continue
            
            color = COLORES_TIPO_SECCION.get(tipo, '#888888')
            
            hover_text = []
            for idx, row in df_tipo.iterrows():
                estado_nombre = ESTADOS.get(row.get('ID_ENTIDAD'), 'N/A')
                prioridad = row.get('PRIORIDAD_MOVILIZACION', 0)
                competitividad = row.get('COMPETITIVIDAD', 0)
                
                text = f"<b>{tipo.replace('_', ' ')}</b><br>Estado: {estado_nombre}<br>"
                
                if nivel == 'SECCION':
                    seccion = row.get('SECCION', 'N/A')
                    text += f"Sección: {seccion}<br>"
                
                text += (
                    f"<br>"
                    f"Prioridad: {prioridad:.1f}<br>"
                    f"Competitividad: {competitividad:.1f}"
                )
                hover_text.append(text)
            
            fig.add_trace(go.Choroplethmapbox(
                geojson=json.loads(df_tipo.to_json()),
                locations=df_tipo['id'],
                z=[1] * len(df_tipo),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                marker_line_width=0.2 if nivel == 'SECCION' else 0,
                marker_line_color='rgba(50,50,50,0.3)' if nivel == 'SECCION' else 'rgba(0,0,0,0)',
                marker_opacity=opacidad,
                hovertext=hover_text,
                hoverinfo='text',
                name=tipo.replace('_', ' ').title()
            ))
        
        estado_nombre = ESTADOS.get(estado_id, 'Nacional') if estado_id else 'Nacional'
        
        fig.update_layout(
            mapbox_style='carto-positron',
            mapbox_zoom=zoom_level,
            mapbox_center=center_coords,
            title={
                'text': f'<b>Clasificación Estratégica de Secciones</b><br><sub>{estado_nombre} - {nivel}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            margin={'r': 0, 't': 60, 'l': 0, 'b': 0},
            paper_bgcolor='#f8f9fa',
            font={'family': 'Arial, sans-serif'},
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#333333",
                borderwidth=1,
                title=dict(text='Tipo de Sección', font=dict(size=12, family='Arial Black'))
            ),
            height=700
        )
        
        return fig

    def _crear_mapa_tendencia(self, gdf_plot, metrica, nivel, estado_id, opacidad=0.65):
        center_coords = COORDS_ESTADOS.get(estado_id, {'lat': 23.6345, 'lon': -102.5528})
        zoom_level = 7 if estado_id else 4
        
        gdf_plot['id'] = gdf_plot.index
        
        # CORRECCIÓN: Sanitizar solo niveles agregados
        if nivel in ['MUNICIPIO', 'DISTRITO_FEDERAL', 'DISTRITO_LOCAL']:
            gdf_plot = self._sanitize_for_json(gdf_plot)
        
        partido = metrica.replace('TENDENCIA_HISTORICA_', '')
        
        fig = go.Figure()
        
        orden_tendencias = [
            'CRECIMIENTO_SOSTENIDO', 'EXPANSION_RAPIDA', 'CRECIMIENTO',
            'RECUPERACION', 'VOLATIL', 'DECLIVE', 'DECLIVE_SOSTENIDO',
            'DECLIVE_RAPIDO', 'AUGE_Y_CAIDA'
        ]
        
        tendencias_presentes = gdf_plot[metrica].dropna().unique()
        tendencias_presentes = [t for t in orden_tendencias if t in tendencias_presentes]
        
        for tendencia in tendencias_presentes:
            df_tend = gdf_plot[gdf_plot[metrica] == tendencia]
            
            if len(df_tend) == 0:
                continue
            
            color = COLORES_TENDENCIA.get(tendencia, '#888888')
            
            hover_text = []
            votos_2024_col = f'{partido}_2024'
            votos_2018_col = f'{partido}_2018'
            
            for idx, row in df_tend.iterrows():
                estado_nombre = ESTADOS.get(row.get('ID_ENTIDAD'), 'N/A')
                votos_2024 = row.get(votos_2024_col, 0)
                votos_2018 = row.get(votos_2018_col, 0)
                cambio = votos_2024 - votos_2018
                
                text = f"<b>{tendencia.replace('_', ' ')}</b><br>Estado: {estado_nombre}<br>"
                
                if nivel == 'SECCION':
                    seccion = row.get('SECCION', 'N/A')
                    text += f"Sección: {seccion}<br>"
                
                text += (
                    f"<br>"
                    f"Votos 2024: {votos_2024:,.0f}<br>"
                    f"Votos 2018: {votos_2018:,.0f}<br>"
                    f"Cambio: {cambio:+,.0f}"
                )
                hover_text.append(text)
            
            fig.add_trace(go.Choroplethmapbox(
                geojson=json.loads(df_tend.to_json()),
                locations=df_tend['id'],
                z=[1] * len(df_tend),
                colorscale=[[0, color], [1, color]],
                showscale=False,
                marker_line_width=0.2 if nivel == 'SECCION' else 0,
                marker_line_color='rgba(50,50,50,0.3)' if nivel == 'SECCION' else 'rgba(0,0,0,0)',
                marker_opacity=opacidad,
                hovertext=hover_text,
                hoverinfo='text',
                name=tendencia.replace('_', ' ').title()
            ))
        
        estado_nombre = ESTADOS.get(estado_id, 'Nacional') if estado_id else 'Nacional'
        
        fig.update_layout(
            mapbox_style='carto-positron',
            mapbox_zoom=zoom_level,
            mapbox_center=center_coords,
            title={
                'text': f'<b>Tendencia Histórica: {partido}</b><br><sub>{estado_nombre} - {nivel}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            margin={'r': 0, 't': 60, 'l': 0, 'b': 0},
            paper_bgcolor='#f8f9fa',
            font={'family': 'Arial, sans-serif'},
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#333333",
                borderwidth=1,
                title=dict(text='Tendencia', font=dict(size=12, family='Arial Black'))
            ),
            height=700
        )
        
        return fig

    def generar_estadisticas(self, nivel='SECCION', estado_id=None, metrica=None):
        df = self.agregar_por_nivel(nivel, estado_id)
        
        if len(df) == 0:
            return {}
        
        lista_col = 'LISTA_NOMINAL_2024'
        
        total_votos = 0
        if 'TOTAL_VOTOS_2024' in df.columns:
            total_votos = df['TOTAL_VOTOS_2024'].sum()
        else:
            party_cols = [f"{p}_2024" for p in base_parties if f"{p}_2024" in df.columns]
            if party_cols:
                total_votos = df[party_cols].sum().sum()
        
        participacion_col = 'PARTICIPACION_PCT'
        abstencion_col = 'ABSTENCION_PCT'
        
        partido_nombre = None
        votos_partido = None
        if metrica and metrica.endswith('_2024') and metrica != lista_col:
            if metrica in df.columns:
                votos_partido = df[metrica].sum()
                partido_nombre = metrica.replace('_2024', '')
        
        stats = {
            'total_votos': total_votos,
            'total_lista_nominal': df[lista_col].sum() if lista_col in df.columns else 0,
            'participacion_promedio': df[participacion_col].mean() if participacion_col in df.columns else 0,
            'abstencion_promedio': df[abstencion_col].mean() if abstencion_col in df.columns else 0,
            'num_secciones': len(df),
            'partido_seleccionado': partido_nombre,
            'votos_partido': votos_partido,
        }
        
        stats['votos_oposicion'] = df['COALICION_OPOSITORA'].sum() if 'COALICION_OPOSITORA' in df.columns else 0
        stats['votos_oficialismo'] = df['COALICION_OFICIALISTA'].sum() if 'COALICION_OFICIALISTA' in df.columns else 0
        stats['votos_mc'] = df['MC_TOTAL'].sum() if 'MC_TOTAL' in df.columns else 0
        
        party_cols_2024 = [f"{p}_2024" for p in base_parties if f"{p}_2024" in df.columns]
        if party_cols_2024:
            votos_por_partido = {col.replace('_2024', ''): df[col].sum() for col in party_cols_2024}
            votos_por_partido = {k: v for k, v in votos_por_partido.items() if v > 0}
            
            if votos_por_partido:
                partido_ganador = max(votos_por_partido, key=votos_por_partido.get)
                partidos_ordenados = sorted(votos_por_partido.items(), key=lambda x: x[1], reverse=True)
                
                stats['ganador_partido'] = partido_ganador
                stats['votos_ganador_partido'] = votos_por_partido[partido_ganador]
                
                if len(partidos_ordenados) > 1:
                    stats['segundo_partido'] = partidos_ordenados[1][0]
                    stats['votos_segundo_partido'] = partidos_ordenados[1][1]
                    stats['margen_victoria_partido'] = stats['votos_ganador_partido'] - stats['votos_segundo_partido']
                    stats['margen_victoria_pct_partido'] = (
                        stats['margen_victoria_partido'] / stats['votos_ganador_partido'] * 100
                    ) if stats['votos_ganador_partido'] > 0 else 0
                else:
                    stats['segundo_partido'] = 'N/A'
                    stats['votos_segundo_partido'] = 0
                    stats['margen_victoria_partido'] = 0
                    stats['margen_victoria_pct_partido'] = 0
        
        votos_coaliciones = {
            'Coalición Opositora': stats['votos_oposicion'],
            'Coalición Oficialista': stats['votos_oficialismo'],
            'Movimiento Ciudadano': stats['votos_mc']
        }
        
        ganador = max(votos_coaliciones, key=votos_coaliciones.get)
        segundo = sorted(votos_coaliciones.items(), key=lambda x: x[1], reverse=True)[1]
        
        stats['ganador'] = ganador
        stats['votos_ganador'] = votos_coaliciones[ganador]
        stats['segundo_lugar'] = segundo[0]
        stats['votos_segundo'] = segundo[1]
        stats['margen_victoria'] = stats['votos_ganador'] - stats['votos_segundo']
        stats['margen_victoria_pct'] = (stats['margen_victoria'] / stats['votos_ganador'] * 100) if stats['votos_ganador'] > 0 else 0
        
        return stats

# ============================================================================
# APLICACIÓN DASH
# ============================================================================
def crear_app(visualizador):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
    
    metricas_disponibles = []
    
    if 'PARTICIPACION_PCT' in visualizador.df.columns:
        metricas_disponibles.append('PARTICIPACION_PCT')
    if 'ABSTENCION_PCT' in visualizador.df.columns:
        metricas_disponibles.append('ABSTENCION_PCT')
    
    for year in years:
        for partido in base_parties:
            col = f'{partido}_{year}'
            if col in visualizador.df.columns:
                metricas_disponibles.append(col)
    
    for partido in base_parties:
        for prefix in ['RETENCION_', 'CRECIMIENTO_AJUSTADO_', 'SHARE_2024_', 'SHARE_2018_', 
                       'CAMBIO_SHARE_', 'VOTOS_GANADOS_', 'VOTOS_PERDIDOS_', 'VOLATILIDAD_HISTORICA_']:
            col = f'{prefix}{partido}'
            if col in visualizador.df.columns:
                metricas_disponibles.append(col)
    
    metricas_competitividad = [
        'MARGEN_VICTORIA_2024', 'COMPETITIVIDAD', 'VOTOS_PARA_VOLTEAR',
        'PRIORIDAD_MOVILIZACION', 'VOLATILIDAD_TOTAL', 'NEP_2024', 'HHI_2024'
    ]
    for metrica in metricas_competitividad:
        if metrica in visualizador.df.columns:
            metricas_disponibles.append(metrica)
    
    metricas_base = [
        'LISTA_NOMINAL_2024', 'TOTAL_VOTOS_2024', 'LISTA_NOMINAL_2018',
        'TOTAL_VOTOS_2018', 'LISTA_NOMINAL_2012', 'TOTAL_VOTOS_2012',
        'CANDIDATO_NO_REGISTRADO_2024', 'VOTOS_NULOS_2024'
    ]
    for metrica in metricas_base:
        if metrica in visualizador.df.columns:
            metricas_disponibles.append(metrica)
    
    if 'TIPO_SECCION_ESTRATEGICA' in visualizador.df.columns:
        metricas_disponibles.append('TIPO_SECCION_ESTRATEGICA')
    
    for partido in base_parties:
        col = f'TENDENCIA_HISTORICA_{partido}'
        if col in visualizador.df.columns:
            metricas_disponibles.append(col)
    
    for coalicion in coaliciones:
        col = f'{coalicion}_2024'
        if col in visualizador.df.columns:
            metricas_disponibles.append(col)
    
    metricas_disponibles.append('Por partidos')
    
    niveles_disponibles = [{'label': '🔹 Sección Electoral', 'value': 'SECCION'}]
    
    if 'DISTRITO_FEDERAL' in visualizador.gdf.columns:
        niveles_disponibles.append({'label': '🔸 Distrito Federal', 'value': 'DISTRITO_FEDERAL'})
    if 'DISTRITO_LOCAL' in visualizador.gdf.columns:
        niveles_disponibles.append({'label': '🔶 Distrito Local', 'value': 'DISTRITO_LOCAL'})
    if 'MUNICIPIO' in visualizador.gdf.columns:
        niveles_disponibles.append({'label': '🔷 Municipio', 'value': 'MUNICIPIO'})
    
    # ========================================================================
    # LAYOUT
    # ========================================================================
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-map-marked-alt me-3"),
                    "Visualizador Electoral México"
                ], className="text-center text-white py-4 mb-0",
                   style={'backgroundColor': '#2C3E50', 'borderRadius': '10px'}),
                html.P(
                    "Sistema Profesional de Análisis Geoespacial Electoral",
                    className="text-center text-muted mt-2 mb-4",
                    style={'fontSize': '18px', 'fontWeight': '300'}
                )
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-sliders-h me-2"),
                        html.B("Panel de Control")
                    ], style={'backgroundColor': '#34495E', 'color': 'white'}),
                    dbc.CardBody([
                        html.Label([
                            html.I(className="fas fa-map-marker-alt me-2"),
                            "Estado:"
                        ], className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='dropdown-estado',
                            options=[{'label': nombre, 'value': id_est} 
                                     for id_est, nombre in sorted(ESTADOS.items(), key=lambda x: x[1])],
                            value=1,
                            clearable=False,
                            placeholder="Selecciona un estado...",
                            className="mb-3"
                        ),
                        
                        html.Label([
                            html.I(className="fas fa-layer-group me-2"),
                            "Nivel Territorial:"
                        ], className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='dropdown-nivel',
                            options=niveles_disponibles,
                            value='SECCION',
                            clearable=False,
                            className="mb-3"
                        ),
                        
                        html.Label([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Métrica a Visualizar:"
                        ], className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id='dropdown-metrica',
                            options=[{'label': m, 'value': m} for m in sorted(metricas_disponibles)],
                            value='PARTICIPACION_PCT',
                            clearable=False,
                            className="mb-3",
                            optionHeight=50,
                            style={'fontSize': '12px'}
                        ),
                        
                        html.Hr(),
                        
                        html.Label([
                            html.I(className="fas fa-adjust me-2"),
                            "Transparencia del Mapa:"
                        ], className="fw-bold mb-2"),
                        dcc.Slider(
                            id='slider-opacidad',
                            min=0.3,
                            max=1.0,
                            step=0.05,
                            value=0.65,
                            marks={
                                0.3: {'label': '30%', 'style': {'fontSize': '10px'}},
                                0.65: {'label': '65%', 'style': {'fontSize': '10px'}},
                                1.0: {'label': '100%', 'style': {'fontSize': '10px'}}
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),
                        html.Small("↓ Menor = Ver nombres del mapa base", 
                                  className="text-muted d-block mb-3"),
                        
                        html.Hr(),
                        
                        dbc.Checklist(
                            options=[{
                                "label": html.Span([
                                    html.I(className="fas fa-trophy me-2"),
                                    "Mostrar Mapa de Ganadores"
                                ]),
                                "value": 1
                            }],
                            value=[],
                            id="switch-ganador",
                            switch=True,
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        dbc.Button([
                            html.I(className="fas fa-sync-alt me-2"),
                            "Actualizar Vista"
                        ], id="btn-actualizar", color="primary", className="w-100 mb-2"),
                        
                        dbc.Button([
                            html.I(className="fas fa-download me-2"),
                            "Descargar Imagen"
                        ], id="btn-descargar", color="success", className="w-100"),
                        
                        html.Hr(),
                        
                        dbc.Alert([
                            html.I(className="fas fa-info-circle me-2"),
                            html.Small([
                                "💡 Ajusta la ",
                                html.B("opacidad"),
                                " para ver nombres de municipios y lugares del mapa base."
                            ], style={'fontSize': '11px'})
                        ], color="info", className="mb-0 mt-2", style={'padding': '8px'})
                    ])
                ], className="shadow-sm")
            ], width=12, lg=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-globe-americas me-2"),
                        html.B("Visualización Geográfica")
                    ], style={'backgroundColor': '#34495E', 'color': 'white'}),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-mapa",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id='mapa-principal',
                                    config={
                                        'scrollZoom': True,
                                        'displayModeBar': True,
                                        'displaylogo': False,
                                        'doubleClick': 'reset',
                                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                        'toImageButtonOptions': {
                                            'format': 'png',
                                            'filename': 'mapa_electoral',
                                            'height': 1200,
                                            'width': 1600,
                                            'scale': 2
                                        }
                                    },
                                    style={'height': '70vh'}
                                )
                            ]
                        )
                    ], style={'padding': '0'})
                ], className="shadow-sm")
            ], width=12, lg=9)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-pie me-2"),
                        html.B("Estadísticas Generales")
                    ], style={'backgroundColor': '#2C3E50', 'color': 'white'}),
                    dbc.CardBody([
                        dbc.Row(id='panel-estadisticas')
                    ])
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Distribución por Partido/Coalición 2024", className="fw-bold"),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='grafico-partidos', style={'height': '400px'})
                        )
                    ])
                ], className="shadow-sm")
            ], width=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Participación Electoral", className="fw-bold"),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='grafico-participacion', style={'height': '400px'})
                        )
                    ])
                ], className="shadow-sm")
            ], width=12, lg=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    html.I(className="fas fa-info-circle me-2"),
                    "Visualizador Electoral México 2024 | Datos Oficiales INE"
                ], className="text-center text-muted small")
            ], width=12)
        ])
        
    ], fluid=True, style={'backgroundColor': '#ECF0F1', 'minHeight': '100vh', 'padding': '20px'})
    
    # ========================================================================
    # CALLBACKS
    # ========================================================================
    
    @app.callback(
        [Output('mapa-principal', 'figure'),
         Output('panel-estadisticas', 'children'),
         Output('grafico-partidos', 'figure'),
         Output('grafico-participacion', 'figure')],
        [Input('btn-actualizar', 'n_clicks')],
        [State('dropdown-estado', 'value'),
         State('dropdown-nivel', 'value'),
         State('dropdown-metrica', 'value'),
         State('switch-ganador', 'value'),
         State('slider-opacidad', 'value')]
    )
    def actualizar_visualizacion(n_clicks, estado_id, nivel, metrica, mostrar_ganador, opacidad):
        estado_id = None if estado_id == 0 else estado_id
        mostrar_ganador = len(mostrar_ganador) > 0 if mostrar_ganador else False
        
        fig_mapa = visualizador.crear_mapa(
            metrica=metrica,
            nivel=nivel,
            estado_id=estado_id,
            mostrar_ganador=mostrar_ganador,
            opacidad=opacidad
        )
        
        stats = visualizador.generar_estadisticas(nivel, estado_id, metrica)
        panel_stats = crear_panel_estadisticas(stats)
        fig_partidos = crear_grafico_partidos(visualizador, nivel, estado_id)
        fig_participacion = crear_grafico_participacion(visualizador, nivel, estado_id)
        
        return fig_mapa, panel_stats, fig_partidos, fig_participacion
    
    return app

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def crear_panel_estadisticas(stats):
    if not stats:
        return [dbc.Col([html.P("No hay datos disponibles", className="text-muted")], width=12)]
    
    colores_ganador = {
        'PAN': '#0066CC', 'PRI': '#FF0000', 'PRD': '#FFD700',
        'MORENA': '#A0522D', 'PVEM': '#00A651', 'PT': '#C1272D', 'MC': '#FF6600'
    }
    
    color_ganador_partido = colores_ganador.get(stats.get('ganador_partido', ''), '#9370DB')
    
    cards = [
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-vote-yea me-2"), "Total de Votos"], 
                           className="text-muted mb-2"),
                    html.H3(f"{stats['total_votos']:,.0f}", className="text-primary mb-0"),
                    html.Small(f"Lista Nominal: {stats['total_lista_nominal']:,.0f}", 
                              className="text-muted")
                ])
            ], className="border-start border-primary border-4 shadow-sm")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-chart-line me-2"), "Participación"], 
                           className="text-muted mb-2"),
                    html.H3(f"{stats['participacion_promedio']:.1f}%", className="text-success mb-0"),
                    html.Small(f"Abstención: {stats['abstencion_promedio']:.1f}%", 
                              className="text-muted")
                ])
            ], className="border-start border-success border-4 shadow-sm")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-trophy me-2"), "Ganador (Partido)"], 
                           className="text-muted mb-2"),
                    html.H4(stats.get('ganador_partido', 'N/A'), className="mb-1", 
                           style={'color': color_ganador_partido, 'fontWeight': 'bold'}),
                    html.Small(f"{stats.get('votos_ganador_partido', 0):,.0f} votos", 
                              className="text-muted")
                ])
            ], className="border-start border-4 shadow-sm", 
               style={'borderColor': color_ganador_partido + ' !important'})
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-balance-scale me-2"), "Margen de Victoria"], 
                           className="text-muted mb-2"),
                    html.H3(f"{stats.get('margen_victoria_pct_partido', 0):.1f}%", 
                           className="text-warning mb-0"),
                    html.Small(f"{stats.get('margen_victoria_partido', 0):,.0f} votos de diferencia", 
                              className="text-muted")
                ])
            ], className="border-start border-warning border-4 shadow-sm")
        ], width=12, md=6, lg=3, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-medal me-2"), "Segundo Lugar (Partido)"], 
                           className="text-muted mb-2"),
                    html.H5(stats.get('segundo_partido', 'N/A'), className="mb-1"),
                    html.Small(f"{stats.get('votos_segundo_partido', 0):,.0f} votos", 
                              className="text-muted")
                ])
            ], className="border-start border-secondary border-4 shadow-sm")
        ], width=12, md=6, lg=4, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-map-marked me-2"), "Unidades Territoriales"], 
                           className="text-muted mb-2"),
                    html.H3(f"{stats['num_secciones']:,}", className="text-info mb-0"),
                    html.Small("Total analizadas", className="text-muted")
                ])
            ], className="border-start border-info border-4 shadow-sm")
        ], width=12, md=6, lg=4, className="mb-3"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6([html.I(className="fas fa-bullseye me-2"), 
                            "Métrica Seleccionada" if stats.get('partido_seleccionado') else "Coaliciones"], 
                           className="text-muted mb-2"),
                    html.H5(stats.get('partido_seleccionado', 'Ver distribución'), 
                           className="mb-1",
                           style={'color': colores_ganador.get(stats.get('partido_seleccionado', ''), '#666')}),
                    html.Small(f"{stats.get('votos_partido', 0):,.0f} votos" 
                              if stats.get('votos_partido') else "Análisis general", 
                              className="text-muted")
                ])
            ], className="border-start border-dark border-4 shadow-sm")
        ], width=12, md=12, lg=4, className="mb-3"),
    ]
    
    return cards

def crear_grafico_partidos(visualizador, nivel, estado_id):
    df = visualizador.agregar_por_nivel(nivel, estado_id)
    
    if len(df) == 0:
        return go.Figure().add_annotation(
            text="No hay datos disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    votos_data = {}
    for partido in base_parties:
        col = f'{partido}_2024'
        if col in df.columns:
            votos = df[col].sum()
            if votos > 0:
                votos_data[partido] = votos
    
    coaliciones_cols = [
        ('PAN_PRI_PRD_2024', 'PAN-PRI-PRD'),
        ('PAN_PRI_2024', 'PAN-PRI'),
        ('PAN_PRD_2024', 'PAN-PRD'),
        ('PRI_PRD_2024', 'PRI-PRD'),
        ('PVEM_PT_MORENA_2024', 'PVEM-PT-MORENA'),
        ('PVEM_PT_2024', 'PVEM-PT'),
        ('PVEM_MORENA_2024', 'PVEM-MORENA'),
        ('PT_MORENA_2024', 'PT-MORENA')
    ]
    
    total_votos_partidos = sum(votos_data.values()) if votos_data else 1
    
    for col, nombre in coaliciones_cols:
        if col in df.columns:
            votos = df[col].sum()
            if votos > 0 and (votos / total_votos_partidos * 100) >= 0.5:
                votos_data[nombre] = votos
    
    votos_data = dict(sorted(votos_data.items(), key=lambda x: x[1], reverse=True))
    
    if not votos_data:
        return go.Figure().add_annotation(
            text="No hay datos de partidos/coaliciones disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    total_votos = sum(votos_data.values())
    
    for entidad, votos in votos_data.items():
        porcentaje = (votos / total_votos * 100) if total_votos > 0 else 0
        color = COLORES_PARTIDOS.get(entidad, COLORES_PARTIDOS.get(entidad.replace('-', '_'), '#888888'))
        
        fig.add_trace(go.Bar(
            x=[entidad],
            y=[votos],
            name=entidad,
            marker_color=color,
            text=[f"{votos:,.0f}<br>({porcentaje:.1f}%)"],
            textposition='outside',
            hovertemplate=f'<b>{entidad}</b><br>Votos: %{{y:,.0f}}<br>Porcentaje: {porcentaje:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': '<b>Distribución de Votos por Partido y Coalición 2024</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial'}
        },
        xaxis_title="Partido / Coalición",
        yaxis_title="Número de Votos",
        showlegend=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font={'family': 'Arial, sans-serif'},
        margin={'t': 60, 'b': 100, 'l': 80, 'r': 20},
        hovermode='x unified',
        annotations=[
            dict(
                text="<i>Nota: Solo se muestran coaliciones con ≥0.5% del total de votos</i>",
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
    )
    
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor='#e0e0e0')
    
    return fig

def crear_grafico_participacion(visualizador, nivel, estado_id):
    df = visualizador.agregar_por_nivel(nivel, estado_id)
    participacion_col = 'PARTICIPACION_PCT'
    
    if len(df) == 0 or participacion_col not in df.columns:
        return go.Figure().add_annotation(
            text="No hay datos de participación disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    participacion = df[participacion_col].mean()
    abstencion = 100 - participacion
    
    total_votos = 0
    if 'TOTAL_VOTOS_2024' in df.columns:
        total_votos = df['TOTAL_VOTOS_2024'].sum()
    
    lista_col = 'LISTA_NOMINAL_2024'
    total_lista = df[lista_col].sum() if lista_col in df.columns else 0
    votos_abstencion = total_lista - total_votos if total_lista > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=['Participación', 'Abstención'],
        values=[participacion, abstencion],
        hole=0.5,
        marker=dict(colors=['#28a745', '#dc3545']),
        textinfo='label+percent',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>Porcentaje: %{percent}<br>Votos: %{customdata:,.0f}<extra></extra>',
        customdata=[total_votos, votos_abstencion]
    ))
    
    fig.add_annotation(
        text=f"<b>{participacion:.1f}%</b><br><sub>Participación</sub>",
        x=0.5, y=0.5,
        font_size=20,
        showarrow=False
    )
    
    fig.update_layout(
        title={
            'text': '<b>Participación Electoral 2024</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial'}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font={'family': 'Arial, sans-serif'},
        margin={'t': 50, 'b': 50, 'l': 20, 'r': 20}
    )
    
    return fig

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
if __name__ == '__main__':
    import sys
    
    # ========================================================================
    # ⚙️ CONFIGURACIÓN
    # ========================================================================
    
    CSV_PATH = os.getenv('CSV_PATH', str(BASE_DIR / 'data' / 'maestro_electoral_con_metricascorregido.csv'))
    SHP_PATH = os.getenv('SHP_PATH', str(BASE_DIR / 'data' / 'SECCION.shp'))
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8050))
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    
    # ========================================================================
    
    print("="*80)
    print("🗳️  VISUALIZADOR ELECTORAL MÉXICO")
    print("="*80)
    
    print("\n🔍 Verificando archivos...")
    
    archivos_ok = True
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: No se encuentra el archivo CSV")
        print(f"   Ruta buscada: {CSV_PATH}")
        archivos_ok = False
    else:
        tamano_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
        print(f"✅ CSV encontrado ({tamano_mb:.2f} MB)")
    
    if not os.path.exists(SHP_PATH):
        print(f"❌ ERROR: No se encuentra el archivo SHP")
        print(f"   Ruta buscada: {SHP_PATH}")
        archivos_ok = False
    else:
        print(f"✅ SHP encontrado")
        base_path = os.path.splitext(SHP_PATH)[0]
        for ext in ['.shx', '.dbf', '.prj']:
            archivo = base_path + ext
            if os.path.exists(archivo):
                print(f"   ✓ {ext}")
            else:
                print(f"   ⚠️  Falta {ext}")
    
    if not archivos_ok:
        print("\n❌ NO SE PUEDEN CARGAR LOS ARCHIVOS")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("⏳ CARGANDO DATOS...")
    print("="*80 + "\n")
    
    try:
        visualizador = VisualizadorElectoral(CSV_PATH, SHP_PATH)
        
        print(f"✅ DATOS CARGADOS EXITOSAMENTE")
        print("="*80)
        print(f"📊 Registros en CSV: {len(visualizador.df):,}")
        print(f"🗺️  Geometrías en SHP: {len(visualizador.gdf):,}")
        print(f"🔗 Registros fusionados: {len(visualizador.df_merged):,}")
        
        if 'ID_ENTIDAD' in visualizador.df.columns:
            estados = visualizador.df['ID_ENTIDAD'].nunique()
            print(f"📍 Estados encontrados: {estados}")
        
        if 'SECCION' in visualizador.df.columns:
            secciones = visualizador.df['SECCION'].nunique()
            print(f"🔢 Secciones únicas: {secciones:,}")
        
        if 'TOTAL_VOTOS_2024' in visualizador.df.columns:
            total_votos = visualizador.df['TOTAL_VOTOS_2024'].sum()
            print(f"🗳️  Total de votos 2024: {total_votos:,.0f}")
        
        niveles = ['SECCION']
        if 'DISTRITO_FEDERAL' in visualizador.gdf.columns:
            niveles.append('DISTRITO_FEDERAL')
        if 'DISTRITO_LOCAL' in visualizador.gdf.columns:
            niveles.append('DISTRITO_LOCAL')
        if 'MUNICIPIO' in visualizador.gdf.columns:
            niveles.append('MUNICIPIO')
        print(f"\n🎯 Niveles disponibles: {', '.join(niveles)}")
        
        print("\n" + "="*80)
        print("🚀 INICIANDO SERVIDOR WEB...")
        print("="*80)
        
        app = crear_app(visualizador)
        server = app.server
        
        print(f"\n{'='*80}")
        print(f"✨ ¡APLICACIÓN LISTA!")
        print(f"{'='*80}")
        print(f"\n🌐 Abre tu navegador en:")
        print(f"   👉 http://{HOST}:{PORT}")
        print(f"\n💡 Funcionalidades:")
        print(f"   • Visualización por estado y nivel territorial")
        print(f"   • Mapas de calor por métrica")
        print(f"   • Mapa de ganadores por partido")
        print(f"   • Control de opacidad (slider)")
        print(f"   • Sin líneas blancas en niveles agregados (buffer+unary_union)")
        print(f"   • Bordes optimizados según nivel")
        print(f"   • Hover simplificado (Estado + Sección si aplica)")
        print(f"   • Gráficos complementarios")
        print(f"   • Descarga de imágenes HD")
        print(f"\n⏹️  Para detener: Ctrl + C")
        print(f"{'='*80}\n")
        
        app.run(
            debug=DEBUG,
            host=HOST,
            port=PORT
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR AL CARGAR ARCHIVOS:")
        print(f"   {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("👋 APLICACIÓN CERRADA")
    print("="*80)