### TRABAJO FINAL ###
## SHIRLEY GUANOCHANGA ##

#Variables y población objetivo: 
#variable_clave = "serv_hig"
#poblacion_objetivo = "sexo == 'Mujer'

#Vamos a importar las librerias con las que vamos a trabajar 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
# Para visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold
import numpy as np

# Cargar los datos originales
datos = pd.read_csv("sample_endi_model_10p.txt", sep=";")

# Filtrar los datos para incluir solo niñas con sexo femenino y que tengan valores válidos en la columna 'serv_hig'
dat_mujer_serv_hig = datos[(datos['sexo'] == 'Mujer') & (datos['serv_hig'].isin(['Alcantarillado', 'Excusado/pozo', 'Letrina/no tiene']))]
conteo_serv_hig = dat_mujer_serv_hig['serv_hig'].value_counts()

# Resultados
print("Conteo de niñas por categoría de 'serv_hig':")
print(conteo_serv_hig)


# Eliminación de filas con valores no finitos en las columnas especificadas
columnas_con_nulos = ['dcronica', 'region', 'n_hijos', 'tipo_de_piso', 'espacio_lavado', 'categoria_seguridad_alimentaria', 'quintil', 'categoria_cocina', 'categoria_agua', 'serv_hig']
datos_nuevos = datos.dropna(subset=columnas_con_nulos)

# Comprobación
print("N valores no finitos después de la eliminación:")
print(datos_nuevos.isna().sum())

# Convertir la variable categórica serv_hig en binaria
datos_nuevos['serv_hig_binario'] = datos_nuevos['serv_hig'].apply(lambda x: 1 if x == 'Alcantarillado' else 0)

# Filtrar para incluir solo niñas de sexo femenino 
datos_mujer_serv_hig = datos_nuevos[(datos_nuevos['sexo'] == 'Mujer') & (datos_nuevos['serv_hig_binario'] == 1)]

#  variables relevantes
variables = ['n_hijos', 'region', 'sexo', 'condicion_empleo', 'serv_hig_binario']

# Filtrar los datos para las variables seleccionadas y eliminar filas con valores nulos en esas variables
for i in variables:
    datos_mujer_serv_hig = datos_mujer_serv_hig[~datos_mujer_serv_hig[i].isna()]

# Agrupación de datos por sexo, tipo de servicio de higiene y conteo de niñas en cada grupo
conteo_ninas_por_servicio_higiene = datos_mujer_serv_hig.groupby(["sexo", "serv_hig_binario"]).size()
print("Conteo de niños por categoría de 'serv_hig':")
print(conteo_ninas_por_servicio_higiene)

# Variables categóricas y numéricas
variables_categoricas = ['region', 'sexo', 'condicion_empleo']
variables_numericas = ['n_hijos']

# Transformador para estandarizar las variables numéricas
transformador = StandardScaler()
datos_escalados = datos_limpios.copy()
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])


# Convertir las variables categóricas en variables dummy
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años']]
y = datos_dummies["serv_hig_binario"]

# Pesos asociados a cada observación
weights = datos_dummies['fexp_nino']

# Datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')


# Variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

# Modelo de regresión logística
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#PREGUNTA 3:cuando ejecutamos el modelo solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento, 
#podemos examinar el coeficiente correspondiente en el resumen del modelo de regresión logística.

coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Tabla pivote para visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
predictions_class = (predictions > 0.5).astype(int)
comparacion = (predictions_class == y_test)

# Número de folds para la validación cruzada
kf = KFold(n_splits=100)
accuracy_scores = []  
df_params = pd.DataFrame()  

# Iterar sobre cada fold
for train_index, test_index in kf.split(X_train):
    # Division los datos en conjuntos de entrenamiento y prueba para este fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustar un modelo de regresión 
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Predicciones
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

# Precisión promedio de la validación cruzada
mean_accuracy = np.mean(accuracy_scores)
print(f"Precisión promedio de validación cruzada: {mean_accuracy}")

# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)
print(precision_promedio)
#Resp Final: Cuando se utiliza el conjunto de datos filtrado, la precisión promedio del modelo disminuye en 
#comparación con el valor previo. 

# HISTOGRAMA 
plt.hist(accuracy_scores, bins=30, edgecolor='blue')
plt.axvline(precision_promedio, color='black', linestyle='dashed', linewidth=2)

# Texto para la precisión promedio
plt.text(precision_promedio - 0.1, plt.ylim()[1] - 0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Histograma de los coeficientes para la variable "n_hijos"
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

media_coeficientes_n_hijos = np.mean(df_params["n_hijos"])
plt.axvline(media_coeficientes_n_hijos, color='blue', linestyle='dashed', linewidth=2)
plt.text(media_coeficientes_n_hijos - 0.1, plt.ylim()[1] - 0.1, f'Media de los coeficientes: {media_coeficientes_n_hijos:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()