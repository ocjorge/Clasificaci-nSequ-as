# Modelo de Clasificación de Sequías

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/yourusername/repo)
[![Documentation](https://img.shields.io/badge/Docs-Passing-success.svg)](https://github.com/yourusername/repo)
[![Issues](https://img.shields.io/github/issues/yourusername/repo.svg)](https://github.com/yourusername/repo/issues)

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
print(f"Probabilidades: {prediccion[0]}")
```

## 🧮 Metodología

El proyecto utiliza las siguientes técnicas para obtener un modelo robusto:

1. **Preprocesamiento**:
   - Normalización de variables con StandardScaler
   - Codificación de la variable objetivo mediante LabelEncoder

2. **Manejo del desbalance**:
   - Técnica SMOTE para generar muestras sintéticas de clases minoritarias
   - Ponderación de clases inversamente proporcional a su frecuencia

3. **Arquitectura del modelo**:
   - Red neuronal feed-forward con 3 capas ocultas
   - Regularización L2 para evitar sobreajuste
   - Dropout para mejorar generalización
   - Función de activación softmax para clasificación multiclase

4. **Estrategia de entrenamiento**:
   - Validación cruzada estratificada (5-fold)
   - Early stopping para detener entrenamiento cuando no hay mejoras
   - Reducción adaptativa del learning rate
   - Batch size pequeño (16) adecuado para dataset pequeño

## 📈 Resultados

El modelo genera:

- Informe de clasificación con precision, recall y F1-score para cada clase
- Matriz de confusión visualizada
- Gráficos de la evolución de accuracy y loss durante el entrenamiento
- Modelo guardado en formato .h5 para uso futuro
- Archivos auxiliares (scaler.pkl y label_encoder.pkl) para preprocesar nuevos datos

## ⚙️ Personalización

Puedes modificar varios aspectos del modelo:

1. **Arquitectura de la red**:
   - Ajustar número de capas y neuronas en `create_model()`
   - Modificar tasa de dropout o factor de regularización L2

2. **Hiperparámetros**:
   - Cambiar batch_size, epochs, learning_rate
   - Ajustar los parámetros de early_stopping y reduce_lr

3. **Balanceo**:
   - Modificar parámetros de SMOTE
   - Ajustar manualmente los class_weights

4. **Selección de características**:
   - Implementar métodos de selección como PCA, RFE o feature importance

## 📞 Contacto

Para consultas o soporte relacionado con este modelo:

- [Tu nombre/organización]
- [Tu email/contacto]
- [Enlaces relevantes]

---

*Nota: Este modelo está diseñado para propósitos educativos y de investigación. Para aplicaciones operativas en gestión hídrica real, debe ser validado por expertos en hidrología y adaptado a las condiciones locales específicas.*
