# Modelo de Clasificación de Sequías

Este proyecto implementa un modelo de red neuronal para clasificar niveles de sequía basado en datos históricos y características hidrológicas. El sistema está diseñado para manejar datasets desbalanceados donde ciertas categorías de sequía son más frecuentes que otras.

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Dataset](#estructura-del-dataset)
- [Instalación y Requisitos](#instalación-y-requisitos)
- [Uso del Modelo](#uso-del-modelo)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Personalización](#personalización)
- [Contacto](#contacto)

## 📝 Descripción del Proyecto

Este proyecto utiliza técnicas de aprendizaje profundo para clasificar niveles de sequía en diferentes categorías: LIGERA, MODERADA, INTENSA, EXTREMA y SEVERA. El sistema está diseñado específicamente para manejar:

- Datos desbalanceados donde algunas categorías tienen muy pocas muestras
- Múltiples variables hidrológicas, meteorológicas y geográficas
- Relaciones complejas entre variables

## 📊 Estructura del Dataset

El modelo utiliza el dataset "cinco.csv" que contiene 75 registros y 28 columnas:

- **Variable objetivo**: Species (LIGERA, MODERADA, INTENSA, EXTREMA, SEVERA)
- **Variables predictoras**: 27 características que incluyen:
  - Datos temporales (AÑO)
  - Datos regionales (REGION)
  - Indicadores hidrológicos (Dispo, Déficit, etc.)
  - Información sobre acuíferos y precipitaciones
  - Datos demográficos y de infraestructura
  - Mediciones relacionadas con presas y volúmenes de agua

La distribución de clases presenta un desbalance significativo:
- LIGERA: 45 muestras (60%)
- MODERADA: 16 muestras (21.3%)
- INTENSA: 9 muestras (12%)
- EXTREMA: 3 muestras (4%)
- SEVERA: 2 muestras (2.7%)

## 🛠️ Instalación y Requisitos

Para ejecutar este proyecto necesitas:

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
```

Requisitos:
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- imbalanced-learn (imblearn)
- pandas
- numpy
- matplotlib

## 🚀 Uso del Modelo

### Entrenamiento del modelo

```bash
python modelo_clasificacion_sequia.py
```

### Predicción con el modelo entrenado

```python
import tensorflow as tf
import numpy as np
import pickle

# Cargar el modelo
model = tf.keras.models.load_model('modelo_clasificacion_sequia.h5')

# Cargar el escalador y el codificador
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Datos nuevos (ajustar según tus características)
nuevos_datos = np.array([[...]])  # Insertar valores para las 27 características

# Preprocesar
nuevos_datos_scaled = scaler.transform(nuevos_datos)

# Predecir
prediccion = model.predict(nuevos_datos_scaled)
clase_predicha = label_encoder.inverse_transform([np.argmax(prediccion[0])])[0]
print(f"Clase predicha: {clase_predicha}")
print(f"Probabilidades: {
