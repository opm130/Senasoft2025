import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENTRENAMIENTO DE MODELO PERSONALIZADO - SENASOFT 2025")
print("="*70)

# ============================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================

print("\n📂 Paso 1: Cargando dataset...")

df = pd.read_csv('dataset.csv', sep=';', encoding='utf-8')
print(f"✓ Dataset cargado: {len(df)} registros")

# Limpieza de datos
df_clean = df.copy()
df_clean['Edad'].fillna(df_clean['Edad'].median(), inplace=True)
df_clean['Comentario'].fillna('', inplace=True)

# Filtrar solo registros con comentarios válidos
df_clean = df_clean[df_clean['Comentario'].str.len() > 10].copy()
print(f"✓ Registros con comentarios válidos: {len(df_clean)}")

print(f"\n📊 Distribución de clases:")
print(df_clean['Categoría del problema'].value_counts())

# ============================================
# 2. PREPROCESAMIENTO DE TEXTO
# ============================================

print("\n" + "="*70)
print("📝 Paso 2: Preprocesamiento de texto")
print("="*70)

import re
import unicodedata

def limpiar_texto(texto):
    """
    Limpia y normaliza el texto
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Remover acentos
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    
    # Remover caracteres especiales, mantener solo letras y espacios
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    
    # Remover espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

# Aplicar limpieza
df_clean['Comentario_Limpio'] = df_clean['Comentario'].apply(limpiar_texto)
print("✓ Texto limpio y normalizado")

# Mostrar ejemplos
print("\n🔍 Ejemplos de limpieza:")
for i in range(3):
    print(f"\nOriginal: {df_clean['Comentario'].iloc[i]}")
    print(f"Limpio:   {df_clean['Comentario_Limpio'].iloc[i]}")

# ============================================
# 3. CREACIÓN DE FEATURES (CARACTERÍSTICAS)
# ============================================

print("\n" + "="*70)
print("🔧 Paso 3: Creación de características (features)")
print("="*70)

# Feature 1: Vectorización TF-IDF del texto
print("\n1. Vectorización TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=2000,           # Top 2000 palabras más importantes
    ngram_range=(1, 3),          # Unigramas, bigramas y trigramas
    min_df=2,                    # Palabra debe aparecer al menos 2 veces
    max_df=0.8,                  # Palabra no debe aparecer en más del 80% de docs
    stop_words='spanish'         # Remover palabras comunes del español
)

X_text = vectorizer.fit_transform(df_clean['Comentario_Limpio'])
print(f"   ✓ Matriz TF-IDF creada: {X_text.shape}")

# Feature 2: Características adicionales (metadata)
print("\n2. Características adicionales...")

# Longitud del comentario
df_clean['longitud_comentario'] = df_clean['Comentario'].str.len()

# Número de palabras
df_clean['num_palabras'] = df_clean['Comentario'].str.split().str.len()

# Edad normalizada
df_clean['edad_normalizada'] = (df_clean['Edad'] - df_clean['Edad'].mean()) / df_clean['Edad'].std()

# Features categóricas (one-hot encoding)
df_clean['es_rural'] = df_clean['Zona rural']
df_clean['sin_internet'] = (df_clean['Acceso a internet'] == 0).astype(int)
df_clean['sin_atencion_previa'] = (df_clean['Atención previa del gobierno'] == 0).astype(int)

# Crear matriz de features adicionales
features_adicionales = df_clean[[
    'longitud_comentario', 
    'num_palabras',
    'edad_normalizada',
    'es_rural',
    'sin_internet',
    'sin_atencion_previa'
]].values

print(f"   ✓ Features adicionales creadas: {features_adicionales.shape}")

# Combinar features de texto y metadata
from scipy.sparse import hstack, csr_matrix

X_combined = hstack([X_text, csr_matrix(features_adicionales)])
print(f"\n✓ Features totales combinadas: {X_combined.shape}")

# ============================================
# 4. PREPARACIÓN PARA ENTRENAMIENTO
# ============================================

print("\n" + "="*70)
print("📊 Paso 4: Preparación de conjuntos de entrenamiento y prueba")
print("="*70)

# Target variable (lo que queremos predecir)
y_categoria = df_clean['Categoría del problema']

# División estratificada (mantiene proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, 
    y_categoria, 
    test_size=0.2,           # 80% entrenamiento, 20% prueba
    random_state=42,
    stratify=y_categoria     # Mantener proporción de clases
)

print(f"✓ Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"✓ Conjunto de prueba: {X_test.shape[0]} muestras")

print(f"\n📊 Distribución en entrenamiento:")
print(y_train.value_counts())

# ============================================
# 5. ENTRENAMIENTO DE MÚLTIPLES MODELOS
# ============================================

print("\n" + "="*70)
print("🤖 Paso 5: Entrenamiento de múltiples modelos")
print("="*70)

modelos = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Linear SVM': LinearSVC(C=1.0, max_iter=2000, random_state=42)
}

resultados = {}

for nombre, modelo in modelos.items():
    print(f"\n{'='*70}")
    print(f"🔄 Entrenando: {nombre}")
    print('='*70)
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # Predecir en conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Validación cruzada (más robusto)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
    
    resultados[nombre] = {
        'modelo': modelo,
        'accuracy_test': accuracy,
        'accuracy_cv_mean': cv_scores.mean(),
        'accuracy_cv_std': cv_scores.std(),
        'y_pred': y_pred
    }
    
    print(f"   Precisión en prueba: {accuracy*100:.2f}%")
    print(f"   Validación cruzada (5-fold): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    print(f"\n   📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))

# ============================================
# 6. SELECCIÓN DEL MEJOR MODELO
# ============================================

print("\n" + "="*70)
print("🏆 Paso 6: Selección del mejor modelo")
print("="*70)

# Encontrar mejor modelo por validación cruzada
mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['accuracy_cv_mean'])
mejor_modelo = resultados[mejor_modelo_nombre]['modelo']
mejor_accuracy = resultados[mejor_modelo_nombre]['accuracy_test']

print(f"\n🥇 Mejor modelo: {mejor_modelo_nombre}")
print(f"   Precisión en prueba: {mejor_accuracy*100:.2f}%")
print(f"   Validación cruzada: {resultados[mejor_modelo_nombre]['accuracy_cv_mean']*100:.2f}%")

# Comparación de todos los modelos
print(f"\n📊 Comparación de modelos:")
print(f"{'Modelo':<20} {'Precisión Prueba':<20} {'Validación Cruzada':<20}")
print("-"*60)
for nombre in resultados:
    acc_test = resultados[nombre]['accuracy_test']
    acc_cv = resultados[nombre]['accuracy_cv_mean']
    print(f"{nombre:<20} {acc_test*100:>18.2f}% {acc_cv*100:>18.2f}%")

# ============================================
# 7. MATRIZ DE CONFUSIÓN
# ============================================

print("\n" + "="*70)
print("📈 Paso 7: Análisis de errores - Matriz de Confusión")
print("="*70)

y_pred_mejor = resultados[mejor_modelo_nombre]['y_pred']
cm = confusion_matrix(y_test, y_pred_mejor)

# Visualizar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=mejor_modelo.classes_,
            yticklabels=mejor_modelo.classes_)
plt.title(f'Matriz de Confusión - {mejor_modelo_nombre}')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✓ Matriz de confusión guardada: matriz_confusion.png")

# ============================================
# 8. IMPORTANCIA DE CARACTERÍSTICAS
# ============================================

print("\n" + "="*70)
print("🔍 Paso 8: Análisis de importancia de características")
print("="*70)

if mejor_modelo_nombre in ['Random Forest', 'Gradient Boosting']:
    # Para modelos basados en árboles
    importancias = mejor_modelo.feature_importances_
    
    # Top 20 features más importantes
    indices_top = np.argsort(importancias)[-20:]
    
    # Obtener nombres de features (palabras del vocabulario)
    feature_names = vectorizer.get_feature_names_out().tolist() + [
        'longitud_comentario', 'num_palabras', 'edad_normalizada',
        'es_rural', 'sin_internet', 'sin_atencion_previa'
    ]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importancias[indices_top])
    plt.yticks(range(20), [feature_names[i] for i in indices_top])
    plt.xlabel('Importancia')
    plt.title(f'Top 20 Características más Importantes - {mejor_modelo_nombre}')
    plt.tight_layout()
    plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico de importancia guardado: importancia_features.png")
    
    print("\n🔝 Top 10 características más importantes:")
    for i, idx in enumerate(indices_top[-10:][::-1], 1):
        print(f"   {i}. {feature_names[idx]}: {importancias[idx]:.4f}")

elif mejor_modelo_nombre == 'Logistic Regression':
    # Para regresión logística, los coeficientes indican importancia
    print("\n📊 Palabras clave por categoría:")
    feature_names = vectorizer.get_feature_names_out()
    
    for i, categoria in enumerate(mejor_modelo.classes_):
        coefs = mejor_modelo.coef_[i]
        top_indices = np.argsort(coefs)[-10:]
        
        print(f"\n   {categoria}:")
        for idx in top_indices[::-1]:
            if idx < len(feature_names):
                print(f"      - {feature_names[idx]}: {coefs[idx]:.4f}")

# ============================================
# 9. PRUEBAS CON EJEMPLOS NUEVOS
# ============================================

print("\n" + "="*70)
print("🧪 Paso 9: Pruebas con ejemplos nuevos")
print("="*70)

def predecir_nuevo(texto, edad=30, zona_rural=0, sin_internet=0, sin_atencion=0):
    """
    Predice la categoría de un nuevo comentario
    """
    # Limpiar texto
    texto_limpio = limpiar_texto(texto)
    
    # Vectorizar texto
    texto_vectorizado = vectorizer.transform([texto_limpio])
    
    # Crear features adicionales
    longitud = len(texto)
    num_palabras = len(texto.split())
    edad_norm = (edad - df_clean['Edad'].mean()) / df_clean['Edad'].std()
    
    features_extra = np.array([[longitud, num_palabras, edad_norm, zona_rural, sin_internet, sin_atencion]])
    
    # Combinar features
    X_nuevo = hstack([texto_vectorizado, csr_matrix(features_extra)])
    
    # Predecir
    prediccion = mejor_modelo.predict(X_nuevo)[0]
    
    # Probabilidades (si el modelo las soporta)
    if hasattr(mejor_modelo, 'predict_proba'):
        probabilidades = mejor_modelo.predict_proba(X_nuevo)[0]
        return prediccion, probabilidades
    else:
        return prediccion, None

# Ejemplos de prueba
ejemplos_nuevos = [
    "el hospital no tiene suficientes médicos y los pacientes esperan mucho",
    "necesitamos más profesores en la escuela primaria",
    "hay mucha basura en las calles y nadie la recoge",
    "la delincuencia está muy alta y necesitamos más policía",
    "faltan computadores en el colegio para enseñar tecnología",
    "el agua del río está contaminada y huele mal"
]

print("\n🔮 Predicciones en ejemplos nuevos:\n")
for ejemplo in ejemplos_nuevos:
    prediccion, probs = predecir_nuevo(ejemplo)
    print(f"Comentario: '{ejemplo}'")
    print(f"   → Categoría predicha: {prediccion}")
    
    if probs is not None:
        print(f"   → Confianza:")
        for i, categoria in enumerate(mejor_modelo.classes_):
            print(f"      {categoria}: {probs[i]*100:.1f}%")
    print("-" * 70)

# ============================================
# 10. GUARDAR MODELO ENTRENADO
# ============================================

print("\n" + "="*70)
print("💾 Paso 10: Guardando modelo entrenado")
print("="*70)

# Guardar el mejor modelo
joblib.dump(mejor_modelo, f'modelo_{mejor_modelo_nombre.replace(" ", "_").lower()}.pkl')
print(f"✓ Modelo guardado: modelo_{mejor_modelo_nombre.replace(' ', '_').lower()}.pkl")

# Guardar vectorizador
joblib.dump(vectorizer, 'vectorizador_tfidf.pkl')
print("✓ Vectorizador guardado: vectorizador_tfidf.pkl")

# Guardar estadísticas para normalización
estadisticas = {
    'edad_mean': df_clean['Edad'].mean(),
    'edad_std': df_clean['Edad'].std(),
    'categorias': mejor_modelo.classes_.tolist()
}
joblib.dump(estadisticas, 'estadisticas_modelo.pkl')
print("✓ Estadísticas guardadas: estadisticas_modelo.pkl")

# Guardar reporte completo
with open('reporte_entrenamiento.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("REPORTE DE ENTRENAMIENTO DEL MODELO\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Datos de entrenamiento: {len(df_clean)} registros\n")
    f.write(f"Features totales: {X_combined.shape[1]}\n\n")
    
    f.write("MEJOR MODELO:\n")
    f.write(f"  Nombre: {mejor_modelo_nombre}\n")
    f.write(f"  Precisión en prueba: {mejor_accuracy*100:.2f}%\n")
    f.write(f"  Validación cruzada: {resultados[mejor_modelo_nombre]['accuracy_cv_mean']*100:.2f}%\n\n")
    
    f.write("COMPARACIÓN DE MODELOS:\n")
    f.write("-"*70 + "\n")
    for nombre in resultados:
        f.write(f"{nombre}:\n")
        f.write(f"  Precisión: {resultados[nombre]['accuracy_test']*100:.2f}%\n")
        f.write(f"  Val. cruzada: {resultados[nombre]['accuracy_cv_mean']*100:.2f}%\n\n")
    
    f.write("="*70 + "\n")
    f.write("REPORTE DE CLASIFICACIÓN DEL MEJOR MODELO:\n")
    f.write("="*70 + "\n\n")
    f.write(classification_report(y_test, y_pred_mejor, zero_division=0))

print("✓ Reporte guardado: reporte_entrenamiento.txt")

# ============================================
# 11. CÓDIGO PARA USAR EL MODELO
# ============================================

print("\n" + "="*70)
print("📝 Paso 11: Código para usar el modelo entrenado")
print("="*70)

codigo_uso = """
# CÓDIGO PARA USAR EL MODELO ENTRENADO
# ====================================

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re
import unicodedata

# Cargar modelo y componentes
modelo = joblib.load('modelo_{}.pkl')
vectorizador = joblib.load('vectorizador_tfidf.pkl')
estadisticas = joblib.load('estadisticas_modelo.pkl')

def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z\\s]', ' ', texto)
    texto = re.sub(r'\\s+', ' ', texto).strip()
    return texto

def predecir_categoria(comentario, edad=30, zona_rural=0, sin_internet=0, sin_atencion=0):
    # Limpiar y vectorizar
    texto_limpio = limpiar_texto(comentario)
    texto_vec = vectorizador.transform([texto_limpio])
    
    # Features adicionales
    longitud = len(comentario)
    num_palabras = len(comentario.split())
    edad_norm = (edad - estadisticas['edad_mean']) / estadisticas['edad_std']
    
    features = np.array([[longitud, num_palabras, edad_norm, zona_rural, sin_internet, sin_atencion]])
    X = hstack([texto_vec, csr_matrix(features)])
    
    # Predecir
    prediccion = modelo.predict(X)[0]
    
    if hasattr(modelo, 'predict_proba'):
        probs = modelo.predict_proba(X)[0]
        return prediccion, dict(zip(estadisticas['categorias'], probs))
    
    return prediccion, None

# EJEMPLO DE USO:
categoria, probabilidades = predecir_categoria(
    "necesitamos más médicos en el hospital",
    edad=45,
    zona_rural=1,
    sin_internet=1
)

print(f"Categoría: {{categoria}}")
print(f"Probabilidades: {{probabilidades}}")
""".format(mejor_modelo_nombre.replace(' ', '_').lower())

with open('usar_modelo.py', 'w', encoding='utf-8') as f:
    f.write(codigo_uso)

print("✓ Código de uso guardado: usar_modelo.py")

# ============================================
# RESUMEN FINAL
# ============================================

print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*70)

print(f"\n🎯 Resultados finales:")
print(f"   ✓ Mejor modelo: {mejor_modelo_nombre}")
print(f"   ✓ Precisión: {mejor_accuracy*100:.2f}%")
print(f"   ✓ Registros entrenados: {len(df_clean)}")
print(f"   ✓ Features totales: {X_combined.shape[1]}")

print(f"\n📁 Archivos generados:")
print(f"   1. modelo_{mejor_modelo_nombre.replace(' ', '_').lower()}.pkl")
print(f"   2. vectorizador_tfidf.pkl")
print(f"   3. estadisticas_modelo.pkl")
print(f"   4. reporte_entrenamiento.txt")
print(f"   5. matriz_confusion.png")
if mejor_modelo_nombre in ['Random Forest', 'Gradient Boosting']:
    print(f"   6. importancia_features.png")
print(f"   7. usar_modelo.py")

print(f"\n💡 Próximos pasos:")
print(f"   1. Revisar el reporte_entrenamiento.txt")
print(f"   2. Usar usar_modelo.py para hacer predicciones")
print(f"   3. Integrar el modelo en la aplicación web")

print("\n" + "="*70)
