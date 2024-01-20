#variable de texto
mi_var = "Hola, Soy Shirley"
print(mi_var)

#lista de números
edad_eco2 =[21,22,23,24,26,22,25]
print(edad_eco2)

#diccionario
libros_fav = {"Las_aventuras_SherlockH" : "Libro1","11_22_63_SK" : "libro2","crimen_y_castigo" : "libro3","cosecha_roja_DH" : "libro4", "El_largo_adiós_RC" : "libro5"}
print(libros_fav)

############
vect_enteros=[12]*8
print(vect_enteros)

vect_flotantes=[8.6]*4
print(vect_flotantes)

diccionario = {"entero" : vect_enteros, "flotante" : vect_flotantes}
print(diccionario)

# Creación de Cadenas
cadena_simple = 'Hola a todos, Espero que esten bien!'
cadena_doble=["Mis libros favoritos son novelas negras y policiacas, pero mi favorito Son las Aventuras de Sherlock Holmes donde se encuentran los cuentos las cinco semillas de naranja y el problema final","¿Qué tipos de libros les gustan?"]
print(cadena_doble)

# Uso de pandas para la lectura de una tabla Excel 
import pandas as pd
imp_sri = pd.read_excel("ventas_SRI (1).xlsx")
print(imp_sri)
