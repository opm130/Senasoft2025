import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# 1. CARGA Y EXPLORACIÓN INICIAL DE DATOS
# ============================================

# Cargar el dataset (ajusta la ruta según tu archivo)
df = pd.read_csv('dataset.csv', sep=';', encoding='utf-8')

print("="*60)
print("ANÁLISIS EXPLORATORIO INICIAL")
print("="*60)
print(f"\nDimensiones del dataset: {df.shape}")
print(f"\nPrimeras filas:")
print(df.head())

print(f"\nInformación general:")
print(df.info())

print(f"\nEstadísticas descriptivas:")
print(df.describe())

print(f"\nValores faltantes por columna:")
print(df.isnull().sum())

# ============================================
# 2. LIMPIEZA DE DATOS
# ============================================

print("\n" + "="*60)
print("LIMPIEZA DE DATOS")
print("="*60)

# Crear una copia para limpieza
df_clean = df.copy()

# 2.1 Manejo de valores faltantes en Edad
edad_media = df_clean['Edad'].median()
df_clean['Edad'].fillna(edad_media, inplace=True)
print(f"\n✓ Edades faltantes rellenadas con la mediana: {edad_media:.0f} años")

# 2.2 Manejo de comentarios vacíos
comentarios_vacios = df_clean['Comentario'].isnull().sum()
df_clean['Comentario'].fillna('Sin comentario', inplace=True)
print(f"✓ Comentarios vacíos encontrados: {comentarios_vacios}")

# 2.3 Normalización de género
df_clean['Género'] = df_clean['Género'].replace({
    'M': 'Masculino',
    'F': 'Femenino',
    'Otro': 'Otro'
})
print(f"✓ Género normalizado")

# 2.4 Conversión de fechas
df_clean['Fecha del reporte'] = pd.to_datetime(df_clean['Fecha del reporte'], format='%d/%m/%Y', errors='coerce')
df_clean['Año'] = df_clean['Fecha del reporte'].dt.year
df_clean['Mes'] = df_clean['Fecha del reporte'].dt.month
print(f"✓ Fechas convertidas correctamente")

# 2.5 Crear columnas categóricas legibles
df_clean['Tiene_Internet'] = df_clean['Acceso a internet'].map({0: 'No', 1: 'Sí'})
df_clean['Atencion_Gobierno'] = df_clean['Atención previa del gobierno'].map({0: 'No', 1: 'Sí'})
df_clean['Es_Zona_Rural'] = df_clean['Zona rural'].map({0: 'No', 1: 'Sí'})

print(f"\n✓ Dataset limpio: {df_clean.shape}")
print(f"✓ Valores faltantes restantes: {df_clean.isnull().sum().sum()}")

# ============================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================

print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO")
print("="*60)

# 3.1 Distribución por categoría
print("\nDistribución por Categoría del Problema:")
print(df_clean['Categoría del problema'].value_counts())

# 3.2 Distribución por urgencia
print("\nDistribución por Nivel de Urgencia:")
print(df_clean['Nivel de urgencia'].value_counts())

# 3.3 Distribución por ciudad
print("\nDistribución por Ciudad:")
print(df_clean['Ciudad'].value_counts())

# 3.4 Análisis de acceso a servicios
print("\nAcceso a Internet:")
print(df_clean['Tiene_Internet'].value_counts())

print("\nAtención Previa del Gobierno:")
print(df_clean['Atencion_Gobierno'].value_counts())

print("\nZona Rural vs Urbana:")
print(df_clean['Es_Zona_Rural'].value_counts())

# ============================================
# 4. DETECCIÓN DE SESGOS ÉTICOS
# ============================================

print("\n" + "="*60)
print("ANÁLISIS ÉTICO - DETECCIÓN DE SESGOS")
print("="*60)

# 4.1 Distribución de género
print("\nDistribución por Género:")
print(df_clean['Género'].value_counts(normalize=True) * 100)

# 4.2 Edad promedio por categoría
print("\nEdad promedio por categoría de problema:")
print(df_clean.groupby('Categoría del problema')['Edad'].mean().round(2))

# 4.3 Acceso a internet por zona
print("\nAcceso a internet por zona:")
print(pd.crosstab(df_clean['Es_Zona_Rural'], df_clean['Tiene_Internet'], normalize='index') * 100)

# 4.4 Atención del gobierno por zona
print("\nAtención previa del gobierno por zona:")
print(pd.crosstab(df_clean['Es_Zona_Rural'], df_clean['Atencion_Gobierno'], normalize='index') * 100)

# ============================================
# 5. PREPARACIÓN PARA MACHINE LEARNING
# ============================================

print("\n" + "="*60)
print("PREPARACIÓN PARA MACHINE LEARNING")
print("="*60)

# 5.1 Filtrar registros con comentarios válidos
df_ml = df_clean[df_clean['Comentario'] != 'Sin comentario'].copy()
print(f"\nRegistros con comentarios válidos: {len(df_ml)}")

# 5.2 Preparar datos para clasificación de categoría
X_text = df_ml['Comentario']
y_categoria = df_ml['Categoría del problema']

# 5.3 Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='spanish')
X_vectorized = vectorizer.fit_transform(X_text)

print(f"✓ Textos vectorizados: {X_vectorized.shape}")

# 5.4 División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_categoria, test_size=0.2, random_state=42, stratify=y_categoria
)

print(f"✓ Conjunto de entrenamiento: {X_train.shape}")
print(f"✓ Conjunto de prueba: {X_test.shape}")

# ============================================
# 6. ENTRENAMIENTO DE MODELOS
# ============================================

print("\n" + "="*60)
print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN")
print("="*60)

# 6.1 Modelo Naive Bayes
print("\n1. Naive Bayes Multinomial:")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_score = nb_model.score(X_test, y_test)

print(f"   Precisión: {nb_score*100:.2f}%")
print("\n   Reporte de clasificación:")
print(classification_report(y_test, nb_pred))

# 6.2 Modelo Random Forest
print("\n2. Random Forest:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_score = rf_model.score(X_test, y_test)

print(f"   Precisión: {rf_score*100:.2f}%")
print("\n   Reporte de clasificación:")
print(classification_report(y_test, rf_pred))

# ============================================
# 7. PREDICCIONES DE EJEMPLO
# ============================================

print("\n" + "="*60)
print("EJEMPLOS DE PREDICCIÓN")
print("="*60)

ejemplos = [
    "necesitamos más médicos en el hospital",
    "las calles están muy sucias y llenas de basura",
    "no hay suficientes profesores en la escuela",
    "hay mucha delincuencia en el barrio"
]

print("\nPredicciones con el mejor modelo (Random Forest):\n")
for ejemplo in ejemplos:
    ejemplo_vectorizado = vectorizer.transform([ejemplo])
    prediccion = rf_model.predict(ejemplo_vectorizado)[0]
    print(f"Comentario: '{ejemplo}'")
    print(f"Categoría predicha: {prediccion}\n")

# ============================================
# 8. GUARDAR DATOS LIMPIOS Y MODELOS
# ============================================

print("\n" + "="*60)
print("GUARDANDO RESULTADOS")
print("="*60)

# Guardar dataset limpio
df_clean.to_csv('dataset_limpio.csv', index=False, encoding='utf-8')
print("✓ Dataset limpio guardado: dataset_limpio.csv")

# Guardar métricas en un archivo
with open('metricas_modelos.txt', 'w', encoding='utf-8') as f:
    f.write("MÉTRICAS DE LOS MODELOS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Naive Bayes - Precisión: {nb_score*100:.2f}%\n\n")
    f.write("Reporte de clasificación:\n")
    f.write(classification_report(y_test, nb_pred))
    f.write("\n" + "="*60 + "\n\n")
    f.write(f"Random Forest - Precisión: {rf_score*100:.2f}%\n\n")
    f.write("Reporte de clasificación:\n")
    f.write(classification_report(y_test, rf_pred))

print("✓ Métricas guardadas: metricas_modelos.txt")

print("\n" + "="*60)
print("PROCESO COMPLETADO")
print("="*60)
print("\nResumen:")
print(f"- Registros procesados: {len(df_clean)}")
print(f"- Categorías: {df_clean['Categoría del problema'].nunique()}")
print(f"- Ciudades: {df_clean['Ciudad'].nunique()}")
print(f"- Mejor modelo: Random Forest ({rf_score*100:.2f}% precisión)")
print("\nArchivos generados:")
print("  1. dataset_limpio.csv")
print("  2. metricas_modelos.txt")