import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Cargar datos
df = pd.read_csv('cinco.csv', encoding='cp1252')

# Preparación de los datos
X = df.drop(['Species'], axis=1)  # Todas las columnas excepto 'Species'
y = df['Species']

# Codificar la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Ver la distribución de clases
print("Distribución de clases original:")
for i, label in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    print(f"{label}: {count} ({count/len(y_encoded)*100:.2f}%)")

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Ver la distribución después de SMOTE
unique, counts = np.unique(y_train_resampled, return_counts=True)
print("\nDistribución después de SMOTE:")
for i, count in zip(unique, counts):
    print(f"{label_encoder.classes_[i]}: {count}")

# Calcular pesos de clase (inverso de la frecuencia)
class_weights = {}
for i, label in enumerate(label_encoder.classes_):
    class_weights[i] = len(y_encoded) / (len(label_encoder.classes_) * np.sum(y_encoded == i))

print("\nPesos de clase:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {class_weights[i]:.2f}")

# Definir callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1
)

# Crear modelo
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Validación cruzada estratificada
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_resampled, y_train_resampled)):
    print(f"\nEntrenando fold {fold+1}/{n_splits}")
    
    X_fold_train, X_fold_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
    y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]
    
    model = create_model()
    
    history = model.fit(
        X_fold_train, y_fold_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_fold_val, y_fold_val),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    fold_histories.append(history.history)
    
    # Evaluar en el conjunto de validación
    val_loss, val_acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
    print(f"Fold {fold+1} - Precisión de validación: {val_acc:.4f}")

# Entrenar modelo final con todos los datos de entrenamiento
final_model = create_model()
history = final_model.fit(
    X_train_resampled, y_train_resampled,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# Evaluar en conjunto de prueba
test_loss, test_acc = final_model.evaluate(X_test, y_test)
print(f"\nPrecisión en conjunto de prueba: {test_acc:.4f}")

# Predecir clases
y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_classes,
    target_names=label_encoder.classes_
))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

# Añadir valores a la matriz
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Graficar historia del entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del Modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del Modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Guardar el modelo
final_model.save('modelo_clasificacion_sequia.h5')
print("\nModelo guardado como 'modelo_clasificacion_sequia.h5'")

# Guardar el escalador y codificador para uso futuro
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Scaler y Label Encoder guardados para uso futuro")
