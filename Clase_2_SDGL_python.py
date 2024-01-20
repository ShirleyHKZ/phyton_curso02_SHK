#variable de texto
var_frase = "Mi nombre es Shirley y estudio economía en la EPN"
print(var_frase)

#lista de números
edad_eco2 =[21,22,23,24,25,22]
print(edad_eco2)

#diccionario
mis_materias = {"eco_2" : "L-M-V","eco_publ" : "J-V","ing_finan" : "V","t_juegos" : "L-M"}
print(mis_materias)

############
vect_ents=[14]*7
print(vect_ents)

vect_flotantes=[9.8]*5
print(vect_flotantes)

diccionario = {"entero" : vect_ents, "flotante" : vect_flotantes}
print(diccionario)

# Creación de Cadenas
cadena_simple = 'Hola a todos, Espero que esten bien!'
cadena_doble=["Voy hablarles un poco sobre mi","Estoy estudio economía","y cursando sexto-séptimo semestre"]
print(cadena_doble)

# Uso de pandas para la lectura de una tabla Excel 
import pandas as pd
imp_sri = pd.read_excel("ventas_SRI (1).xlsx")
print(imp_sri)
