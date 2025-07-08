# Datrix 2
import os
def clc():
    os.system('cls' if os.name == 'nt' else 'clear')
clc()
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from skimage.color import rgb2gray
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage import img_as_float
from PIL import Image
import Funciones as fn
from math import ceil, sqrt, hypot
import io
import tempfile
import matplotlib.animation as animation
from matplotlib.animation import writers
from matplotlib.animation import PillowWriter
from skimage.data import shepp_logan_phantom
import pydicom
from numba import njit
from numba.typed import List
from numba import njit, int32, float64
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import scipy.ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
#import tkinter as tk
#from tkinter import filedialog
from skimage.draw import line, polygon
import streamlit_image_coordinates as sc
from streamlit_drawable_canvas import st_canvas
from PIL import ImageDraw
import pandas as pd
from matplotlib.patches import Rectangle, Ellipse, Polygon
import base64
from io import BytesIO
from scipy.interpolate import UnivariateSpline

icono = Image.open("logodatrix.jpg")
##---------
st.set_page_config(
    page_title="Datrix",
    page_icon=icono
)
##---------
#st.title("🧮 DaTrix")
#titulo_personalizado("🧮 DaTrix", nivel=2, tamaño=56, color="black")
# Función para convertir imagen local a base64
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

#Ruta a tu imagen (ajustá el nombre del archivo)
img_base64 = get_image_base64("logodatrix.jpg")

# Título con logo centrado
st.markdown(
    f"""
    <div style='text-align: center;'>
        <h1 style='display: inline-flex; align-items: center; gap: 10px; font-size: 56px;'>
            <img src='data:image/jpeg;base64,{img_base64}' width='56'/>
            DaTrix
        </h1>
        <p style='font-size: 20px;'>App hecha con amor.</p>
    </div>
    """,
    unsafe_allow_html=True
)
#Una app para operaciones con matrices, procesamiento de imágenes y simulaciones de reconstrucción tomográfica.

#st.write("Una app para operaciones con matrices, procesamiento de imágenes y simulaciones de reconstrucción tomográfica.")

# Menú principal
modulo = st.sidebar.selectbox("Seleccionar módulo", [
    "Álgebra y análisis numérico",
    "Procesamiento de imágenes",
    "Reconstrucción tomográfica",
    "Sobre mí"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### 💙 ¿Te gustó la app?")
st.sidebar.markdown("Apoyá el proyecto con una donación:")
st.sidebar.markdown("[📲 Donar vía MercadoPago](http://link.mercadopago.com.ar/datrix)")
st.sidebar.markdown("[📂 Repositorio en GitHub](https://github.com/David-Basti/datrix)")

match modulo:
    case "Álgebra y análisis numérico":
        #with st.expander("Módulo 1",expanded=True):
        st.title("🔢 Módulo 1: Álgebra y análisis aumérico")
        tabs = st.tabs(["Operaciones con matrices", "Señales y sistemas"])
        #titulo_personalizado("🔢 Módulo 1: Operaciones con matriz única", nivel=2, tamaño=56, color="black")
        with tabs[0]:
            cool, _ =st.columns([1,5])
            #titulo_personalizado("📥 Matriz A", nivel=2, tamaño=42, color="black",centrado=False)
            with cool:
                if st.button("🔄 Reiniciar"):
                    st.session_state['expresion'] = None
                    st.session_state['exp2'] = None
                    st.session_state["B"] = None
                    st.session_state["resulvect"] = None
                    st.session_state["resultadoa"] = None
                    st.session_state["resultado"] = None
                    vale3 = 1
                    resultado = None
                    resultadoa = None
                    resulvect = None
                    resultado2 = None
                    ra = None
                    rv = None
                    operacion = None
                    oper = None
                    st.session_state["operacion"] = None
                    st.session_state["resultado2"] = None
                    st.session_state["mantenerpegado"] = None
                    st.session_state["mantenerpegado2"] = None
                    st.session_state["mantenerpegado3"] = None
                    st.session_state["copiado"] = None
                    texto_A = "1, 2, 3\n4, 5, 6\n7, 8, 9"
                    texto_A1 = "1, 2, 3\n4, 5, 6\n7, 8, 9"
                    texto_B = "1, 4, 7\n2, 5, 8\n3, 6, 9"
            if "copiado" not in st.session_state:
                st.session_state["copiado"] = None
            if "mantenerpegado" not in st.session_state:
                st.session_state["mantenerpegado"] = None
            if "mantenerpegado2" not in st.session_state:
                st.session_state["mantenerpegado2"] = None
            if "mantenerpegado3" not in st.session_state:
                st.session_state["mantenerpegado3"] = None
            col1, col2 =st.columns(2)
            with col2: 
                if st.button("Pegar", key="pegado") and st.session_state["copiado"] is not None:
                    st.session_state["mantenerpegado"] = st.session_state["copiado"]
                if st.session_state["mantenerpegado"] is not None:
                    texto_A = st.session_state["mantenerpegado"]
                else:
                    texto_A = "1, 2, 3\n4, 5, 6\n7, 8, 9"
                st.header("📥 Matriz M")
                A = None
                texto_A = st.text_area("Ingresá la matriz M (por filas, separados por coma)", texto_A)#"1 2 3\n4 5 6\n7 8 9")
                try:
                    filas = texto_A.strip().split("\n")
                    A = []

                    for fila in filas:
                        elementos = fila.strip().split(",")
                        fila_simb = [sp.sympify(elem.strip()) for elem in elementos]
                        A.append(fila_simb)
                
                    A = sp.Matrix(A)
                    st.latex(sp.latex(A))
                
                except:
                    st.error(f"❌ Error al interpretar la matriz M")
                    st.stop()
                
                if "B" not in st.session_state:
                    st.session_state["B"] = texto_A
                if st.session_state["B"] != texto_A:
                    st.session_state["expresion"] = A
                    st.session_state["ex2"] = "M"
                    st.session_state["B"] = texto_A
                if "ex2" not in st.session_state:
                    st.session_state["ex2"] = "M"

                if 'expresion' not in st.session_state:
                    st.session_state['expresion'] = A  # Empieza con la matriz A

            with col1:
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.write("🔢 Elevar por un escalar")
                    expr3 = st.text_input("Escalar (a)", value="1")
                    try:
                        escalar_3 = sp.sympify(expr3)
                    except (sp.SympifyError, TypeError):
                        st.error("⚠️ No se pudo interpretar el escalar simbólicamente.")
                        escalar_3 = 1
                        escalar_3 = sp.sympify(escalar_3)
                    try:
                        escalar_3f = float(escalar_3)
                    except (ValueError, TypeError):
                            st.error("❌ El escalar (a) debe ser un número.")
                            escalar_3f = 1
                    if not escalar_3.is_number:
                            st.error("❌ El escalar (a) debe ser un número real.")
                            escalar_3 = 1
                        
                    if st.button("Elevar"):
                        if escalar_3f == 1:
                            st.session_state['expresion'] = f"({st.session_state['expresion']})"
                            #st.session_state['ex2'] = f"({st.session_state['ex2']})"
                        elif   A.shape[0] != A.shape[1]:
                            st.error("❌ La matriz debe ser cuadrada para poder elevarse a una potencia.")
                            escalar_3 = 1
                        elif escalar_3f < 0 and A.det() == 0:
                            #d = A.det()
                            #if d != 0:
                            st.error("❌ Para exponentes negativos el det(A) debe ser distinto de cero")
                            escalar_3 = 1
                        else:
                            st.session_state['expresion'] = f"({st.session_state['expresion']})**{(escalar_3)}"
                            #st.session_state["ex2"] = f"({st.session_state["ex2"]})^({escalar_3})"
                            #else:
                            #st.session_state['expresion'] = f"({st.session_state['expresion']})**{escalar_3}"
                            #st.session_state['ex2'] = f"({st.session_state['ex2']})**{escalar_3}"
            
                with c2:
                    st.write("🔢 Producto por un escalar")
                    expr1 = st.text_input("Escalar (b)", value="1")
                    try:
                        escalar_1 = sp.sympify(expr1)
                #st.write("Escalar (b) interpretado como:", escalar_1)
                    except (sp.SympifyError, TypeError):
                        st.error("⚠️ No se pudo interpretar el escalar simbólicamente.")
                        escalar_1 = 1
                    if st.button("Multiplicar"):
                        if escalar_1 != 1:
                            st.session_state['expresion'] = f"({st.session_state['expresion']}) * {escalar_1}"
                            #st.session_state['ex2'] = f"({st.session_state['ex2']}) * {escalar_1}"

                with c3:
                    st.write("🔢 Sumar un escalar")
                    expr2=st.text_input("Escalar (c)", value="0")
                    try:
                        escalar_2 = sp.sympify(expr2)
                #st.write("Escalar (c) interpretado como:", escalar_2)
                    except (sp.SympifyError, TypeError):
                        st.error("⚠️ No se pudo interpretar el escalar simbólicamente.")
                        escalar_2 = 0
                    if st.button("Sumar"):
                        if escalar_2 != 0:
                            st.session_state['expresion'] =  f"({st.session_state['expresion']}) + {sp.srepr(escalar_2 * sp.ones(*A.shape))}"
                            #st.session_state['ex2'] =  f"({st.session_state['ex2']}) + {escalar_2}"
            
                #with st.expander("Mostrar escalares"):
                #    with c1:
                #        st.write("(a) se interpretó como:", escalar_3)
                #    with c2:
                #        st.write("(b) se interpretó como:", escalar_1)
                #    with c3:
                #        st.write("(c) se interpretó como:",escalar_2)
            R = None
            try:
            # Evaluar la expresión acumulada
                R = sp.sympify(st.session_state['expresion']).subs(A, A)  # Asumiendo que A es tu matriz original
                #st.write("Resultado de la operación acumulada:")
                with st.container():
                    cl1, cl2 = st.columns(2)
                    with cl2:
                #with st.expander("Matriz luego de operaciones escalares:"):
                        st.latex("Matriz\,post\,operaciones\,escalares")
                        st.latex(sp.latex(R))
                    #with cl1:
                        #st.latex(sp.latex("Expresión acumulada"))
                    #st.write("Expresión acumulada:")
                        #st.latex(sp.latex(sp.sympify(st.session_state['ex2'])))
                        #with st.expander(""):
                        #    st.write("La matriz M puede aparecer como denominador si el exponente es negativo." \
                        #    " Eso es incorrecto. Lo correcto es M^(-n).")
                                
            except Exception as e:
                st.error(f"Hubo un error al calcular: {str(e)}")
            if R is not None:
                A = R
            columna1, columna2 = st.columns(2)

            with columna1:
                st.header("⚙️ Operación")
                operacion = st.radio("Elegí una operación", ["Ninguna", "Rango", "Transpuesta", "Determinante", "Inversa",
                                            "Autovalores y autovectores"])
                resultado = None
                resulvect = None
                resultadoa = None
                ra = None
                rv = None
                if st.button("Calcular",key="Calcular"):
                    if A is not None:
                        match operacion:
                            case "Ninguna":
                                st.session_state["resulvect"] = None
                                resultado = A
                                st.session_state["resultado"] = resultado
                            case "Transpuesta":
                                st.session_state["resulvect"] = None
                                resultado = A.T
                                st.session_state["resultado"] = resultado  # lo guardamos
                            case "Rango":
                                st.session_state["resulvect"] = None
                                resultado = A.rank()
                                st.session_state["resultado"] = sp.sympify(resultado)
                            case "Determinante":
                                st.session_state["resulvect"] = None
                                if A.shape[0] == A.shape[1]:
                                    resultado = A.det()
                                    st.session_state["resultado"] = resultado
                                else:
                                    resultado = ("⚠️ La matriz debe ser cuadrada")
                                    st.session_state["resultado"] = resultado
                            case "Inversa":
                                st.session_state["resulvect"] = None
                                if A.shape[0] == A.shape[1]:
                                    determinante = A.det()
                            #dets = As.det()
                                    if determinante == 0:
                                        resultado = ("La matriz no es inversible")
                                        st.session_state["resultado"] = resultado
                                    else:
                                        resultado = A.inv()
                                        st.session_state["resultado"] = resultado
                                #rs = As.inv()    
                                else:
                                    resultado = ("⚠️ La matriz debe ser cuadrada")
                                    st.session_state["resultado"] = resultado
                            case "Autovalores y autovectores":
                                if A.shape[0] == A.shape[1]:
                                    resultado = None
                                    st.session_state["resultado"] = resultado
                                    ra = A.eigenvals()
                                    rv = A.eigenvects()
                                    resultadoa = {val.evalf(): mult for val, mult in ra.items()}
                                    st.session_state["resultadoa"] = resultadoa
                                    st.session_state["resulvect"] = rv 
                                else:
                                    resultado = ("⚠️ La matriz debe ser cuadrada")
                                    st.session_state["resultado"] = resultado
                    else:
                        st.error("⚠️ Debe cargar una matriz")
            with columna2:
                if st.session_state.get("resultado") is not None and st.session_state.get("resulvect") is None:
                    if isinstance(st.session_state["resultado"], str):
                        st.error(st.session_state["resultado"])
                    else:
                        resultado = st.session_state["resultado"]
                        with st.expander("Resultado", expanded=True):
                            usar_d = st.checkbox("Mostrar resultado con decimales",key="mostrar1")
                            if usar_d:
                                st.latex(sp.latex(resultado.evalf()))
                            else:
                                st.latex(sp.latex(resultado))
                            if st.button("Copiar",key="Copiar1"):
                                if isinstance(resultado, sp.Matrix):
                                    st.session_state["copiado"] = "\n".join(",".join(str(val) for val in fila) for fila in resultado.tolist())
                                elif isinstance(resultado,int):
                                    st.session_state["copiado"] = str(resultado)
                                else:
                                    st.session_state["copiado"] = "\n".join(",".join(str(val) for val in fila) for fila in resultado.tolist())
                                    
                if  st.session_state.get("resulvect") is not None and st.session_state.get("resultadoa") is not None:
                    if isinstance(st.session_state["resultado"], str):
                        st.error(st.session_state["resultado"])
                    else:
                        resultadoa = st.session_state["resultadoa"]
                        rv = st.session_state["resulvect"]
                        with st.expander("Autovalores y autovectores", expanded=True):
                            usar1 = st.checkbox("Mostrar decimales, autovalores", key="Autovalores_dec")
                            usar2 = st.checkbox("Mostrar decimales, autovectores", key="Autovectores_dec")
                            k = 1
                            for val, mult, vectores in rv:
                                val_str = sp.latex(val.evalf()) if usar1 else sp.latex(sp.nsimplify(val))
                                st.latex(f"\\lambda_{k} = {val_str},\\ \\text{{multiplicidad}} = {mult}")
                                vectores_proc = [v.evalf() if usar2 else v for v in vectores]
                                for i, v in enumerate(vectores_proc, start=1):
                                    st.latex(f"\\vec{{v}}_{{{k}}} = {sp.latex(v)}")
                                    k += 1
                            #for val, mult in resultadoa.items():
                            #    val_str = sp.latex(val) if usar1 else sp.latex(sp.nsimplify(val))
                            #    st.latex(f"\\lambda_{k} = {val_str},\\ \\text{{multiplicidad}} = {mult}")
                            #    k += 1
                    #with st.expander("Autovectores", expanded=True):
                            #usar2 = st.checkbox("Mostrar decimales", key="Autovectores_dec")
                            #k = 1
                            #for val, mult, vectores in rv:
                            #    vectores_proc = [v.evalf() if usar2 else v for v in vectores]
                            #    for v in vectores_proc:
                            #        st.latex(f"\\vec{{v}}_{k} = {sp.latex(v)}")
                            #        k += 1
        
            col1, col2 = st.columns(2)

            with col1:
            
                st.header("📥 Matriz A")
                A1 = None
                if st.button("Pegar", key="pegado2") and st.session_state["copiado"] is not None:
                    st.session_state["mantenerpegado2"] = st.session_state["copiado"]
                if st.session_state["mantenerpegado2"] is not None:
                    texto_A1 = st.session_state["mantenerpegado2"]
                else:
                    texto_A1 = "1, 2, 3\n4, 5, 6\n7, 8, 9"
                texto_A1 = st.text_area("Ingresá la matriz A (por filas, separados por coma)", texto_A1)

                try:
                    filas = texto_A1.strip().split("\n")
                    A1 = []
                    for fila in filas:
                        elementos = fila.strip().split(",")
                        fila_simb = [sp.sympify(elem.strip()) for elem in elementos]
                        A1.append(fila_simb)
                    A1 = sp.Matrix(A1)
                    st.latex(sp.latex(A1))
                except:
                    st.error(f"❌ Error al interpretar la matriz A")
                    st.stop()
        
    # ------------------------------
    # Entrada para la matriz B
    # ------------------------------
            with col2:
                st.header("📥 Matriz B")
                B = None
                
                if st.button("Pegar", key="pegado3") and st.session_state["copiado"] is not None:
                    st.session_state["mantenerpegado3"] = st.session_state["copiado"]
                if st.session_state["mantenerpegado3"] is not None:
                    texto_B = st.session_state["mantenerpegado3"]
                else:
                    texto_B = "1, 4, 7\n2, 5, 8\n3, 6, 9"
                texto_B = st.text_area("Ingresá la matriz B (por filas, separados por coma)", texto_B)
                
                try:
                    filas = texto_B.strip().split("\n")
                    B = []
                    for fila in filas:
                        elementos = fila.strip().split(",")
                        fila_simb = [sp.sympify(elem.strip()) for elem in elementos]
                        B.append(fila_simb)
                
                    B = sp.Matrix(B)
                    st.latex(sp.latex(B))
                except:
                    st.error(f"❌ Error al interpretar la matriz B")
                    st.stop()

    # ------------------------------
    # Selección de operación
    # ------------------------------

            resultado2 = None
            if "resultado2" not in st.session_state:
                st.session_state["resultado2"]=None
            if "operacion" not in st.session_state:
                st.session_state["operacion"] = None
            st.header("⚙️ Operación")
            
            kolum1, kolum2, kolum3 = st.columns(3)
            
            with kolum1:
                #SS=st.radio("",["Suma", "Sustracción"],["Q"],["W"])
                    if st.button("A+B",key="A+B"):
                    #if st.session_state["A+B"]:
                        st.session_state["operacion"] = "Suma"
                    if st.button("A-B",key="A-B"):
                    #if st.session_state["A-B"]:
                        st.session_state["operacion"] = "Sustracción"
            with kolum2:
                    #BB=st.radio("",["Producto bin a bin", "Cociente bin a bin"])
                    if st.button("A.*B",key="A.*B"):
                        #if st.session_state["A.*B"]:
                        st.session_state["operacion"] = "Producto bin a bin"
                    if st.button("A./B",key="A./B"):
                    #if st.session_state["A./B"]:
                        st.session_state["operacion"] = "Cociente bin a bin"
            with kolum3:
                    #CC=st.radio("",["Producto matricial", "Conmutador"])
                    if st.button("AB",key="AB"):
                    #if st.session_state["AB"]:
                        st.session_state["operacion"] = "Producto matricial"
                    if st.button("AB-BA",key="AB-BA"):
                    #if st.session_state["AB-BA"]:
                        st.session_state["operacion"] = "Conmutador"
            #operacion = st.sidebar.radio("Elegí una operación", ["Suma", "Sustracción", "Producto matricial",
            #"Producto bin a bin", "Cociente bin a bin"])
            
            if st.session_state["operacion"] is not None:
                st.write(f"🔧 Operación seleccionada: **{st.session_state['operacion']}**")
            if st.button("Calcular"):
                oper = st.session_state["operacion"]
                if oper is None:
                    st.warning("⚠️ Primero seleccioná una operación.")
                else:
                    match oper:
                        case "Suma":
                            if A1.shape != B.shape:
                                st.error("❌ la dim(A) y dim(B) deben ser iguales")
                                resultado2 = None
                            else:
                                resultado2 = A1 + B
                        case "Sustracción":
                            if A1.shape != B.shape:
                                st.error("❌ la dim(A) y dim(B) deben ser iguales")
                                resultado2 = None
                            else:
                                resultado2 = A1 - B
                        case "Producto bin a bin":
                            if A1.shape != B.shape:
                                st.error("❌ la dim(A) y dim(B) deben ser iguales")
                                resultado2 = None
                            else:
                                resultado2 = fn.mult_elementwise(A1, B)
                        case "Cociente bin a bin":
                            if A1.shape != B.shape:
                                st.error("❌ la dim(A) y dim(B) deben ser iguales")
                                resultado2 = None
                            else:
                                hay_cero = any([b == 0 for b in B])
                                if hay_cero:
                                    st.error("❌ No se puede dividir bin a bin porque hay al menos un cero en la matriz B.")
                                    resultado2 = None
                                else:
                                    resultado2 = fn.divide_elementwise(A1, B)
                        case "Producto matricial":
                            if A1.shape[1] != B.shape[0]:
                                st.error("El número de columnas de A debe ser igual al número de filas de B")
                                resultado2 = None
                            else:
                                resultado2 = fn.prox(A1, B)
                        case "Conmutador":
                            if A1.shape[1] == B.shape[0] and B.shape[1] == A1.shape[0]:
                                resultado2 = fn.prox(A1, B) - fn.prox(B, A1)
                            else:
                                st.error("Dimensiones incompatibles")
                                resultado2 = None
                        case _:
                            resultado2 = None

                st.session_state["resultado2"] = resultado2
                #st.session_state["calcular_flag"] = False  # Reseteamos el flag
            #st.button("Calcular",key="Calc2")
            #st.session_state["resultado2"]=resultado2
            if st.session_state.get("resultado2") is not None:
                st.write("Resultado")
                usar_dab = st.checkbox("Mostrar resultado con decimales",key="mostrar2")
                if usar_dab:
                    st.latex(sp.latex(st.session_state["resultado2"].evalf()))
                else:
                    st.latex(sp.latex(st.session_state["resultado2"]))
                if st.button("Copiar",key="Copiar2"):
                                if isinstance(st.session_state["resultado2"],sp.Integer):
                                    st.session_state["copiado"] = str(st.session_state["resultado2"])
                                elif isinstance(st.session_state["resultado2"], sp.Matrix):
                                    st.session_state["copiado"] = "\n".join(",".join(str(val) for val in fila) for fila in st.session_state["resultado2"].tolist())
                                #elif isinstance(resultado2,sp.Integer):
                                #    st.session_state["copiado"] = str(st.session_state["resultado2"])
                                #else:
                                #    st.session_state["copiado"] = "\n".join(" ".join(str(val) for val in fila) for fila in st.session_state["resultado2"].tolist())
            with tabs[1]:
                U = None
                f = None
                opcion = st.radio("Elegí una operación", ["Ajuste polinómico", "Interpolación Spline", "Ajuste exponencial","Resolución general de C·x = b"],horizontal=True,key="opcionsec1")
                # --- Lógica para cada opción ---
                if opcion == "Ajuste polinómico":
                    st.title("📐 Ajuste y resolución de sistemas lineales")
                    U, V, W = fn.cargar_datos_uv_w()

                    if U is not None and V is not None:
                        U = np.array(U)
                        V = np.array(V)
                        #st.write(U,V)
                        if len(U) == 0 or len(V) == 0:
                            st.error("Error: las listas de datos U y V no pueden estar vacías.")
                            st.stop()

                        st.subheader("Cortar la función")
                        xinicial = st.number_input(
                            "Extremo inferior",
                            value=float(0),      # valor por defecto
                            min_value=float(0),  # valor mínimo que se puede elegir
                            max_value=float(len(U)),  # valor máximo que se puede elegir
                            step=len(U) / 100,
                            format="%.4f")
                        xfinal = st.number_input("Extremo superior",
                                                value=float(len(U)),      # valor por defecto
                                                min_value=float(0),  # valor mínimo que se puede elegir
                                                max_value=float(len(U)),  # valor máximo que se puede elegir
                                                step=(len(U)) / 100,
                                                format="%.4f")
                        
                        if xinicial<=xfinal:

                            U = U[int(xinicial):int(xfinal)]
                            V = V[int(xinicial):int(xfinal)]
                            
                            if W is not None:
                                W = W[int(xinicial):int(xfinal)]

                            if len(U) == len(V):
                                usar_W = W is not None and len(W) == len(U)

                                st.subheader("Personalización del gráfico")
                                col1, col2 = st.columns([0.3,0.7])
                                with col1:
                                    grado = st.number_input("Grado del polinomio", min_value=1, max_value=len(U)-1, value=1,step=1)
                                    st.write(f"Grado sugerido {len(U)-1}")
                                    titulo = st.text_input("Título del gráfico", value="V vs U")
                                    xlabel = st.text_input("Etiqueta del eje X", value="U, []")
                                    ylabel = st.text_input("Etiqueta del eje Y", value="V, []")

                                resultados = fn.RLE_polyfit(U=U, V=V, grado=grado, W=W if usar_W else None,
                                    plot=True, titulo=titulo, xlabel=xlabel, ylabel=ylabel, use_weights=True, show_errorbars=True)
                                with col2:
                                    st.pyplot(resultados['fig'])

                                st.markdown("### 📊 Resultados")
                                if grado == 1:
                                    a = resultados['a']
                                    b = resultados['b']
                                    sign_b = '+' if b >= 0 else '-'
                                    b_abs = abs(b)
                                    with st.expander("Ver resultados"): 
                                        st.latex(rf"""
                                            \begin{{aligned}}
                                            y &= {a:.5f}x {sign_b} {b_abs:.5f} \\
                                            a &= {a:.5f} \quad & b &= {b:.5f} \\
                                            R^2 &= {resultados['R']:.5f} \quad & D &= {resultados['D']:.5f} \\
                                            \Delta a &= {resultados['𝛥a']:.5f} \quad & \Delta b &= {resultados['𝛥b']:.5f}
                                            \end{{aligned}}
                                        """)
                                else:
                                    ecuacion = fn.formatear_ecuacion(resultados['coef'])
                                    with st.expander("Ver resultados"):  
                                        st.latex(ecuacion)

                                        coef_str = r" \\ ".join([
                                            fn.formato_coeficiente(resultados['coef'][i], resultados['delta_coef'][i], i, grado)
                                            for i in range(len(resultados['coef']))
                                        ])
                                        st.latex(r"\begin{aligned}" + coef_str + r"\end{aligned}")
                                        st.latex(rf"R^2 = {resultados['R2']:.5f}")
                                        st.latex(rf"D = {resultados['D']:.5f}")
                            else:
                                st.warning("Los vectores U y V deben tener la misma longitud.")
                        else: st.warning("El extremo superior debe ser mayor al inferior")
                    else:
                        st.warning("No hay datos válidos para procesar.")

                elif opcion == "Interpolación Spline":
                    st.title("🔗 Interpolación Spline")
                    U, V, W = fn.cargar_datos_uv_w()

                    if U is not None and V is not None and len(U) == len(V):

                        st.subheader("Cortar la función")
                        xinicial = st.number_input(
                            "Extremo inferior en (x)",
                            value=float(0),      # valor por defecto
                            min_value=float(0),  # valor mínimo que se puede elegir
                            max_value=float(len(U)),  # valor máximo que se puede elegir
                            step=len(U) / 100,
                            format="%.4f")
                        xfinal = st.number_input("Extremo en (x) superior",
                                                value=float(len(U)),      # valor por defecto
                                                min_value=float(0),  # valor mínimo que se puede elegir
                                                max_value=float(len(U)),  # valor máximo que se puede elegir
                                                step=(len(U)) / 100,
                                                format="%.4f")
                        if xinicial<=xfinal:
                        
                            U = U[int(xinicial):int(xfinal)]
                            V = V[int(xinicial):int(xfinal)]
                            if W is not None:
                                W = W[int(xinicial):int(xfinal)]

                            if xinicial == xfinal:
                                st.stop()

                            st.subheader("Configuración del gráfico")
                            col1, col2 = st.columns([0.3, 0.7])
                            with col1:
                                titulo = st.text_input("Título del gráfico", value="Spline de V vs U")
                                xlabel = st.text_input("Etiqueta del eje X", value="U, []")
                                ylabel = st.text_input("Etiqueta del eje Y", value="V, []")
                                suavizado = st.slider("Nivel de suavizado (s)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

                                orden = np.argsort(U)
                                U_ord = U[orden]
                                V_ord = V[orden]
                                if W is not None and len(W)==len(U):
                                    W_ord = W[orden]
                                else:
                                    W_ord = None
                            if len(np.unique(U_ord)) != len(U_ord):
                                st.warning("Hay valores repetidos en U.")
                            else:
                                
                                spline = UnivariateSpline(U_ord, V_ord, w=(1/W_ord if W_ord is not None and len(W)==len(U) else None), s=suavizado)

                                U_interp = np.linspace(U_ord.min(), U_ord.max(), 1000)
                                V_interp = spline(U_interp)

                                ylimit2 = st.number_input(
                                                    "Extremo inferior en (y) visible",
                                                    value=float(min(V_interp)),      # valor por defecto
                                                    min_value=float(min(V_interp)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                                                    max_value=float(max(V_interp)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                                                    step=0.01,
                                                    format="%.4f",key="yspline1")
                                ylimit1 = st.number_input(
                                                    "Extremo superior en (y) visible",
                                                    value=float(max(V_interp)),      # valor por defecto
                                                    min_value=float(min(V_interp)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                                                    max_value=float(max(V_interp)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                                                    step=0.01,
                                                    format="%.4f",key="yspline2")

                                with col2:
                                    fig, ax = plt.subplots()
                                    ax.plot(U, V, 'ro', label='Datos')
                                    ax.plot(U_interp, V_interp, label='Spline', color='blue')
                                    if W is not None and len(W)==len(U):
                                        ax.errorbar(U, V, yerr=W, fmt='none', ecolor='black', capsize=3, label="Errores")


                                    ax.set_title(titulo)
                                    ax.set_xlabel(xlabel)
                                    ax.set_ylabel(ylabel)
                                    plt.ylim([ylimit2,ylimit1])
                                    #ax.legend()
                                    st.pyplot(fig)
                        else: st.warning("El extremo superior debe ser mayor al inferior")
                    else:
                        st.warning("No hay datos válidos para realizar la interpolación.")
                        st.stop()

                elif opcion == "Ajuste exponencial":
                    st.title("📈 Ajuste Exponencial")
                    U, V, W = fn.cargar_datos_uv_w()

                    if U is not None and V is not None:
                        U = np.array(U)
                        V = np.array(V)
                        if len(U) == 0 or len(V) == 0:
                            st.error("Error: las listas de datos U y V no pueden estar vacías.")
                            st.stop()

                        st.subheader("Cortar la función")
                        xinicial = st.number_input(
                            "Extremo inferior",
                            value=float(0),
                            min_value=float(0),
                            max_value=float(len(U)),
                            step=len(U) / 100,
                            format="%.4f",
                            key="xexp1")
                        xfinal = st.number_input(
                            "Extremo superior",
                            value=float(len(U)),
                            min_value=float(0),
                            max_value=float(len(U)),
                            step=len(U) / 100,
                            format="%.4f",
                            key="xexp2")

                        if xinicial <= xfinal:
                            U = U[int(xinicial):int(xfinal)]
                            V = V[int(xinicial):int(xfinal)]
                            if W is not None:
                                W = W[int(xinicial):int(xfinal)]

                            if len(U) == len(V):
                                usar_W = W is not None and len(W) == len(U)

                                st.subheader("Personalización del gráfico")
                                col1, col2 = st.columns([0.3, 0.7])
                                with col1:
                                    titulo = st.text_input("Título del gráfico", value="Ajuste Exponencial de V vs U")
                                    xlabel = st.text_input("Etiqueta del eje X", value="U, []")
                                    ylabel = st.text_input("Etiqueta del eje Y", value="V, []")

                                resultados = fn.RLE_exponential_fit(
                                    U=U, V=V, W=W if usar_W else None,
                                    plot=True, titulo=titulo, xlabel=xlabel, ylabel=ylabel,
                                    use_weights=True, show_errorbars=True
                                )

                                with col2:
                                    if resultados and 'fig' in resultados and resultados['fig'] is not None:
                                        st.pyplot(resultados['fig'])
                                    else:
                                        st.warning("No se pudo generar el gráfico. Verifica que los datos no estén vacíos.")
                                        st.stop()



                                st.markdown("### 📊 Resultados del ajuste exponencial")
                                a = resultados['a']
                                b = resultados['b']
                                delta_a = resultados['𝛥a']
                                delta_b = resultados['𝛥b']
                                R2 = resultados['R2']
                                D = resultados['D']
                                c = resultados['c']
                                delta_c = resultados['𝛥c']

                                sign_b = '+' if b >= 0 else '-'
                                b_abs = abs(b)

                                with st.expander("Ver resultados"):                              
                                    st.latex(rf"""
                                        \begin{{aligned}}
                                        y &= {a:.5f} \cdot e^{{{b:.5f}x}} + {c:.5f} \\
                                        a &= {a:.5f} \quad & \Delta a &= {delta_a:.5f} \\
                                        b &= {b:.5f} \quad & \Delta b &= {delta_b:.5f} \\
                                        c &= {c:.5f} \quad & \Delta c &= {delta_c:.5f} \\
                                        R^2 &= {R2:.5f} \quad & D &= {D:.5f}
                                        \end{{aligned}}
                                    """)
                            else:
                                st.warning("Los vectores U y V deben tener la misma longitud.")
                        else:
                            st.warning("El extremo superior debe ser mayor al inferior.")
                    else:
                        st.warning("No hay datos válidos para procesar.")
                    

                elif opcion == "Resolución general de C·x = b":
                                    
                    st.title("🧮 Resolución de sistemas de ecuaciones lineales")
                    st.markdown("Resolvé sistemas lineales de la forma **C·x = b** con coeficientes numéricos o simbólicos.")

                    archivo = st.file_uploader("Subí un archivo con la matriz C y el vector b", type=["csv", "xlsx"])

                    C = None
                    b = None

                    if archivo is not None:
                        try:
                            if archivo.name.endswith(".csv"):
                                df = pd.read_csv(archivo, header=None)
                            else:
                                df = pd.read_excel(archivo, header=None)

                            if df.shape[1] < 2:
                                st.error("El archivo debe tener al menos dos columnas (C y b).")
                            else:
                                try:
                                    df_sym = df.applymap(lambda x: sp.sympify(str(x)))
                                    C = sp.Matrix(df_sym.iloc[:, :-1].values.tolist())
                                    b = sp.Matrix(df_sym.iloc[:, -1].values.tolist())
                                except Exception as e:
                                    st.error(f"Error al interpretar simbólicamente los datos: {e}")
                        except Exception as e:
                            st.error(f"Error al leer el archivo: {e}")

                    else:
                        st.subheader("Carga manual")
                        C_txt = st.text_area("Matriz C (fila por fila, separado por coma)", "2, 1\n1, 3")
                        b_txt = st.text_area("Vector b (una entrada por fila)", "8\n13")

                        try:
                            C_list = [list(map(sp.sympify, fila.split(','))) for fila in C_txt.strip().split('\n')]
                            b_list = list(map(sp.sympify, b_txt.strip().split('\n')))
                            C = sp.Matrix(C_list)
                            b = sp.Matrix(b_list)
                        except Exception as e:
                            st.error(f"Error al procesar los datos simbólicamente: {e}")

                    if C is not None and b is not None and C.rows == b.rows:
                        try:
                            if C.rows == C.cols:
                                x = C.LUsolve(b)
                                x = sp.Matrix([sp.simplify(xi) for xi in x])
                                st.success("✅ Sistema cuadrado: solución exacta encontrada.")
                            else:
                                st.warning("⚠️ Sistema no cuadrado. Se usará método de mínimos cuadrados.")
                                Ct = C.T
                                normal_matrix = Ct * C
                                normal_rhs = Ct * b
                                try:
                                    x = normal_matrix.inv() * normal_rhs
                                    #x = (C.T * C).inv() * (C.T * b)
                                    x = sp.Matrix([sp.simplify(xi) for xi in x])
                                    st.success("✅ Solución por mínimos cuadrados:")
                                except:
                                    x = None
                                    st.error("❌ No se pudo invertir la matriz normal. El sistema no tiene solución por mínimos cuadrados.")

                            if x is not None:
                                # Mostrar la solución en LaTeX
                                solucion_latex = r"x = \begin{bmatrix}" + r"\\".join([sp.latex(xi) for xi in x]) + r"\end{bmatrix}"
                                st.latex(solucion_latex)

                                # Mostrar el residuo simbólico y su norma
                                residuo = C * x - b
                                residuo = sp.Matrix([sp.simplify(ri) for ri in residuo])
                                error = sp.sqrt(sum([ri**2 for ri in residuo]))
                                st.latex(rf"\text{{Error del residuo: }} \|C \cdot x - b\| = {sp.latex(error)}")

                        except Exception as e:
                            st.error(f"No se pudo resolver el sistema simbólicamente: {e}")
                    else:
                        st.warning("Verificá que C y b estén bien definidos y tengan la misma cantidad de filas.")
                if U is not None:
                    st.subheader("Análisis de curva")
                    if "CurvaFiltrada" not in st.session_state:
                        st.session_state["CurvaFiltrada"] = None
                    usar_curva_filtrada = st.checkbox("Usar curva filtrada", value=False)
                            
                # Para el ajuste polinómico:
                if opcion == "Ajuste polinómico":
                    if U is not None and len(U) == len(V):
                        if xinicial<=xfinal:
                            volu1, volu2 = st.columns([0.3,0.7])
                            resultadospoli = fn.RLE_polyfit2(volu1=volu1,volu2=volu2,U=U, V=V, grado=grado, W=W if usar_W else None,
                                    plot=True, titulo=titulo, xlabel=xlabel, ylabel=ylabel, use_weights=True, show_errorbars=True)
                            umin, umax = np.min(U), np.max(U)
                            escalar = lambda x: 2 * (x - umin) / (umax - umin) - 1
                            f = lambda x: np.polyval(resultadospoli['coef'], escalar(x))
                            #st.subheader("📍 Líneas de referencia")
                            # Slider para dos líneas verticales (x)

                            # 2. Evaluar f(x) en todo el rango posible de x (o en el dominio total de datos)
                            x_full = resultadospoli["U_plot"]#np.linspace(np.min(U), np.max(U), 1000)
                            if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                y_full = st.session_state["CurvaFiltrada"]
                            else:
                                y_full = f(x_full)

                            # 3. Establecer rangos posibles para los sliders
                            x_min_full, x_max_full = np.min(x_full), np.max(x_full)
                            y_min_full, y_max_full = np.min(y_full), np.max(y_full)

                            with volu1:
                                # 4. Sliders (ahora ya tenemos min y max válidos)
                                x_lines = st.slider("Intervalo en X (líneas verticales)", 
                                                    min_value=float(x_min_full), max_value=float(x_max_full), 
                                                    value=(float(x_min_full), float(x_max_full)), 
                                                    step=(x_max_full - x_min_full) / 100,format="%.4f")

                                y_lines = st.slider("Intervalo en Y (líneas horizontales)", 
                                                    min_value=float(y_min_full), max_value=float(y_max_full), 
                                                    value=(float(y_min_full), float(y_max_full)), 
                                                    step=(y_max_full - y_min_full) / 100,format="%.4f")

                            # 5. Evaluar f(x) en intervalo de x_lines
                            #x_eval = np.linspace(x_lines[0], x_lines[1], 1000)
                            #y_eval = f(x_eval)
                            y_max = -np.inf
                            x_max = None
                            y_min = np.inf
                            x_min = None

                            for i in range(len(x_full)):
                                x = x_full[i]
                                y = y_full[i]
                                if x_lines[0] <= x <= x_lines[1] and y_lines[0] <= y <= y_lines[1]:
                                    if y > y_max:
                                        y_max = y
                                        x_max = x
                                    if y < y_min:
                                        y_min = y
                                        x_min = x


                            st.success(f"📈 Dentro del rectángulo:\n\n🔽 Mínimo f(x) = {y_min:.4f} en x = {x_min:.4f}\n🔼 Máximo f(x) = {y_max:.4f} en x = {x_max:.4f}")
                            with volu2:
                                if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                    figpoli,ax =  plt.subplots()
                                    ax.plot(x_full, st.session_state["CurvaFiltrada"], color="b", label="Curva Filtrada")
                                else:
                                    figpoli = resultadospoli["fig"]
                                ax = figpoli.gca()
                                ax.plot(x_min, y_min, 'gv', label='Mín.')
                                ax.plot(x_max, y_max, 'r^', label='Máx.')
                                #else:
                                #    st.warning("⚠️ No hay puntos de la curva dentro del rectángulo definido.")
                                # Mostrar
                                #st.success(f"📉 Mínimo de f(x) = {y_min_abs:.4f} en x = {x_min_abs:.4f}\n📈 Máximo de f(x) = {y_max_abs:.4f} en x = {x_max_abs:.4f}")


                                # Líneas verticales
                                ax.axvline(x=x_lines[0], color='cyan', linestyle='--', label=f'x₁ = {x_lines[0]:.2f}')
                                ax.axvline(x=x_lines[1], color='cyan', linestyle='--', label=f'x₂ = {x_lines[1]:.2f}')

                                # Líneas horizontales
                                ax.axhline(y=y_lines[0], color='y', linestyle='--', label=f'y₁ = {y_lines[0]:.2f}')
                                ax.axhline(y=y_lines[1], color='y', linestyle='--', label=f'y₂ = {y_lines[1]:.2f}')

                                #ax.legend()
                                st.pyplot(figpoli)

                # Para spline:
                elif opcion == "Interpolación Spline":
                    if U is not None and len(U) == len(V):
                        if xinicial<=xfinal:
                            orden = np.argsort(U)
                            U_ord = U[orden]
                            V_ord = V[orden]
                            
                            if len(np.unique(U_ord)) != len(U_ord):
                                pass#st.warnig("Hay valores repetidos en U. Pruebe usar suavizado")
                            else:
                                if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                    f = UnivariateSpline(U_interp,st.session_state["CurvaFiltrada"],s=0)
                                else:
                                    f = spline  # ya es un callable
                                # --- Líneas de referencia ---
                                #st.subheader("📍 Líneas de referencia")

                                # 2. Evaluar f(x) en todo el rango posible de x (o en el dominio total de datos)
                                x_full = U_interp
                                y_full = f(x_full)

                                # 3. Establecer rangos posibles para los sliders
                                x_min_full, x_max_full = min(x_full), max(x_full)
                                y_min_full, y_max_full = min(y_full), max(y_full)

                                volu1, volu2 = st.columns([0.3,0.7])
                                with volu1:
                                    # 4. Sliders (ahora ya tenemos min y max válidos)
                                    x_lines = st.slider("Intervalo en X (líneas verticales)", 
                                                        min_value=float(x_min_full), max_value=float(x_max_full), 
                                                        value=(float(x_min_full), float(x_max_full)), 
                                                        step=(x_max_full - x_min_full) / 100,format="%.4f")

                                    y_lines = st.slider("Intervalo en Y (líneas horizontales)", 
                                                        min_value=float(y_min_full), max_value=float(y_max_full), 
                                                        value=(float(y_min_full), float(y_max_full)), 
                                                        step=float((y_max_full - y_min_full) / 100) / 100,format="%.4f")

                                # 5. Evaluar f(x) en intervalo de x_lines
                                y_max = -np.inf
                                x_max = None
                                y_min = np.inf
                                x_min = None

                                for i in range(len(x_full)):
                                    x = x_full[i]
                                    y = y_full[i]
                                    if x_lines[0] <= x <= x_lines[1] and y_lines[0] <= y <= y_lines[1]:
                                        if y > y_max:
                                            y_max = y
                                            x_max = x
                                        if y < y_min:
                                            y_min = y
                                            x_min = x

                                st.success(f"📈 Dentro del rectángulo:\n\n🔽 Mínimo f(x) = {y_min:.4f} en x = {x_min:.4f}\n🔼 Máximo f(x) = {y_max:.4f} en x = {x_max:.4f}")
                                
                                #else:
                                #    st.warning("⚠️ No hay puntos de la curva dentro del rectángulo definido.")
                                with volu2:
                                    figspline, ax = plt.subplots()
                                    ax.plot(x_min, y_min, 'gv', label='Mín.')
                                    ax.plot(x_max, y_max, 'r^', label='Máx.')
                                    if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                        ax.plot(U_interp,st.session_state["CurvaFiltrada"],label="Spline",color="blue")
                                    else:
                                        ax.plot(U_interp, V_interp, label='Spline', color='blue')
                                    #ax.plot(U, V, 'ro', label='Datos')
                                    ax = figspline.gca()

                                    # Líneas verticales
                                    ax.axvline(x=x_lines[0], color='cyan', linestyle='--', label=f'x₁ = {x_lines[0]:.2f}')
                                    ax.axvline(x=x_lines[1], color='cyan', linestyle='--', label=f'x₂ = {x_lines[1]:.2f}')

                                    # Líneas horizontales
                                    ax.axhline(y=y_lines[0], color='y', linestyle='--', label=f'y₁ = {y_lines[0]:.2f}')
                                    ax.axhline(y=y_lines[1], color='y', linestyle='--', label=f'y₂ = {y_lines[1]:.2f}')

                                    #ax.legend()
                                    st.pyplot(figspline)
                
                if opcion == "Ajuste exponencial":
                    if U is not None and len(U) == len(V):
                        if xinicial <= xfinal:
                            volu1, volu2 = st.columns([0.3, 0.7])


                            f = resultados["f"]
                            x_full = resultados["x_plot"]
                            if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                y_full = st.session_state["CurvaFiltrada"]
                            else:
                                y_full = resultados["y_plot"]

                            # Establecer rangos posibles para los sliders
                            x_min_full, x_max_full = float(min(x_full)), float(max(x_full))
                            y_min_full, y_max_full = float(min(y_full)), float(max(y_full))

                            with volu1:
                                x_lines = st.slider("Intervalo en X (líneas verticales)",
                                                    min_value=x_min_full, max_value=x_max_full,
                                                    value=(x_min_full, x_max_full),
                                                    step=(x_max_full - x_min_full) / 100,
                                                    format="%.4f")

                                y_lines = st.slider("Intervalo en Y (líneas horizontales)",
                                                    min_value=y_min_full, max_value=y_max_full,
                                                    value=(y_min_full, y_max_full),
                                                    step=(y_max_full - y_min_full) / 100,
                                                    format="%.4f")

                            # Buscar mínimos y máximos dentro del rectángulo
                            y_max = -np.inf
                            x_max = None
                            y_min = np.inf
                            x_min = None

                            for i in range(len(x_full)):
                                x = x_full[i]
                                y = y_full[i]
                                if x_lines[0] <= x <= x_lines[1] and y_lines[0] <= y <= y_lines[1]:
                                    if y > y_max:
                                        y_max = y
                                        x_max = x
                                    if y < y_min:
                                        y_min = y
                                        x_min = x

                            st.success(f"📈 Dentro del rectángulo:\n\n🔽 Mínimo f(x) = {y_min:.4f} en x = {x_min:.4f}\n🔼 Máximo f(x) = {y_max:.4f} en x = {x_max:.4f}")

                            with volu2:
                                if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                    figexp,ax =  plt.subplots()
                                    ax.plot(x_full, st.session_state["CurvaFiltrada"], color="b", label="Curva Filtrada")
                                else:
                                    figexp = resultados["fig2"]
                                
                                ax = figexp.gca()
                                ax.plot(x_min, y_min, 'gv', label='Mín.')
                                ax.plot(x_max, y_max, 'r^', label='Máx.')

                                # Líneas verticales
                                ax.axvline(x=x_lines[0], color='cyan', linestyle='--', label=f'x₁ = {x_lines[0]:.2f}')
                                ax.axvline(x=x_lines[1], color='cyan', linestyle='--', label=f'x₂ = {x_lines[1]:.2f}')

                                # Líneas horizontales
                                ax.axhline(y=y_lines[0], color='y', linestyle='--', label=f'y₁ = {y_lines[0]:.2f}')
                                ax.axhline(y=y_lines[1], color='y', linestyle='--', label=f'y₂ = {y_lines[1]:.2f}')

                                st.pyplot(figexp)
                
                if f is not None:
                    curana = st.tabs(["Integración", "Resolución de ecuación", "Transformada de Fourier"])
                    with curana[0]:
                        st.subheader("Integración")
                        # Calcular área
                        # Límites de integración
                        polum1,polum2 = st.columns([0.3,0.7])
                        with polum1:
                            a = st.number_input("Límite inferior de integración (a)", value=float(min(U)), min_value=float(min(U)), max_value=float(max(U)),format="%.4f")
                            b = st.number_input("Límite superior de integración (b)", value=float(max(U)), min_value=float(min(U)), max_value=float(max(U)),format="%.4f")
                            h = st.number_input("Paso de integración (h)", value=0.01, min_value=1e-4, format="%.4f")
                        if h <= abs(b-a):
                            with polum1:
                                tipo_integracion = st.selectbox("Método de integración", ["simpson", "trapecio", "riemann"])
                                if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                    f = UnivariateSpline(x_full,st.session_state["CurvaFiltrada"],s=0)
                                area = fn.integral(f, a, b, h, tipo=tipo_integracion)
                            with polum2:
                                st.markdown(f"### 📐 Área bajo la curva entre {a} y {b}:")
                                st.latex(rf"\int_{{{a}}}^{{{b}}} f(x)\,dx \approx {area:.5f}")
                                if opcion == "Interpolación Spline":
                                    x_plot = U_interp
                                    y_plot = V_interp
                                elif opcion == "Ajuste polinómico":
                                    x_plot = resultadospoli["U_plot"]
                                    y_plot = resultadospoli["V_plot"]
                                elif opcion == "Ajuste exponencial":
                                    x_plot = np.linspace(min(U), max(U), 1000)
                                    y_plot = resultados["f"](x_plot)
                                # --- Mostrar gráfico con área sombreada ---
                                fig_area, ax = plt.subplots()
                                if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                    ax.plot(x_plot, st.session_state["CurvaFiltrada"], color="b", label="Función")
                                else:
                                    ax.plot(x_plot, y_plot, label="Función", color="blue")
                                x_fill = np.linspace(a, b, 500)
                                y_fill = f(x_fill)
                                ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label="Área bajo f(x)")
                                ax.set_title("Área bajo la curva")
                                ax.legend()
                                st.pyplot(fig_area)
                        else:
                            with polum2:
                                st.warning("El paso h es absurdo.")
                    with curana[1]:
                        st.subheader("Resolución de ecuación")
                        col1, col2 = st.columns([0.3,0.7])
                        with col1:
                            p_val = st.number_input("Elegí el valor de f(x) = p", value=0.0, step=0.01, format="%.4f")
                        if opcion == "Interpolación Spline":
                            # Función g(x)
                            def g(x):
                                return f(x) - p_val
                            # Derivada de g(x)
                            dg = f.derivative()  # esta es f'(x)
                        elif opcion == "Ajuste polinómico": 
                            c = np.arange(0,grado+1,1)
                            c = c[::-1]
                            u = sp.symbols("u")
                            h=0
                            for k in range(len(c)):
                                h+=resultadospoli['coef'][k]*u**c[k]
                            gsym = sp.sympify(h-p_val)
                            dgsym = sp.diff(gsym, u)*2 / (umax - umin)
                            #st.latex(gsym)
                            #st.latex(dgsym)
                            g_u = sp.lambdify(u, gsym, "math")   # función evaluable
                            dg_u = sp.lambdify(u, dgsym, "math") # derivada evaluable
                            if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                # Función g(x)
                                def g(x):
                                    return f(x) - p_val
                                # Derivada de g(x)
                                dg = f.derivative()  # esta es f'(x)
                            else:
                                g = lambda x: float(g_u(escalar(x)))
                                dg = lambda x: float(dg_u(escalar(x)))
                        elif "Ajuste exponencial":
                            def g(x):
                                return f(x) - p_val
                            dg = resultados["df"]
                            

                        
                        with col1:
                            a_custom = st.number_input("Límite inferior", value=float(min(U)),format="%.4f")
                            b_custom = st.number_input("Límite superior", value=float(max(U)),format="%.4f")

                            tol = st.number_input("Tolerancia de Newton-Raphson", value=1e-6, format="%.1e")
                            max_iter = st.number_input("Máximo de iteraciones de bisección", value=10)

                        #if st.button("🔧 Buscar raíz"):
                        try:
                            x_sol,_,_ = fn.biner2(g, dg,a_custom, b_custom, max_iter,tol)
                            st.success(f"✅ Solución: x ≈ {x_sol:.6f} tal que f(x) ≈ {p_val:.4f}")

                            if opcion == "Interpolación Spline":
                                x_plot = U_interp
                                y_plot = V_interp
                            elif opcion == "Ajuste polinómico":
                                x_plot = resultadospoli["U_plot"]
                                y_plot = resultadospoli["V_plot"]
                            elif opcion == "Ajuste exponencial":
                                x_plot = np.linspace(min(U), max(U), 1000)
                                y_plot = resultados["f"](x_plot)

                            fig, ax = plt.subplots()
                            if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                ax.plot(x_plot,st.session_state["CurvaFiltrada"],label="Función",color="b")
                            else:
                                ax.plot(x_plot, y_plot, label="Función", color='blue')
                            ax.axhline(p_val, color='gray', linestyle='--', label=f"p = {p_val}")
                            ax.plot(x_sol, f(x_sol), 'go', label=f"x ≈ {x_sol:.4f}")
                            ax.legend()
                            with col2:
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    with curana[2]:
                        st.subheader("Análisis de Fourier")

                        # === 1. Sobremuestreo ===
                        M = 1000
                        #aosdmow
                        # === 2. FFT con zero-padding ===
                        N_fft = 2**14  # largo total con padding (16384 puntos), potencia de 2 para rendimiento
                        V_padded = np.zeros(N_fft)
                        if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                            V_padded[:M] = st.session_state["CurvaFiltrada"]
                        else:
                            V_padded[:M] = y_plot  # colocamos la función interpolada al inicio

                        # Tiempo entre muestras
                        dt = (U.max() - U.min()) / (M - 1)
                        fm = 1 / dt  # frecuencia de muestreo
                        nyquist = fm / 2

                        # === 3. FFT centrada ===
                        fft_vals = np.fft.fft(V_padded)
                        fft_vals_shifted = np.fft.fftshift(fft_vals)
                        EM = np.abs(fft_vals_shifted)

                        # === 4. Eje de frecuencias ===
                        freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))  # Hz
                        freqs_rad = 2 * np.pi * freqs  # rad/s
                        aol1,aol2=st.columns(2)

                        # Inputs para rango de frecuencia en rad/s
                        min_freq = float(freqs_rad.min())
                        max_freq = float(freqs_rad.max())

                        with aol1:
                            freq_start = st.number_input("Frecuencia inicial (rad/s)", min_value=min_freq, max_value=max_freq, value=min_freq, format="%.4f")
                        with aol2:
                            freq_end = st.number_input("Frecuencia final (rad/s)", min_value=min_freq, max_value=max_freq, value=max_freq, format="%.4f")

                        if freq_start >= freq_end:
                            st.error("La frecuencia inicial debe ser menor a la frecuencia final.")
                        else:
                            filtro = (freqs_rad >= freq_start) & (freqs_rad <= freq_end)
                            
                            col1, col2 = st.columns(2)
                            st.markdown("### 🔎 Comparación del espectro de magnitud (FFT centrada)")

                            fig_mag, ax_mag = plt.subplots()
                            ax_mag.plot(freqs_rad[filtro], EM[filtro], color='blue')
                            ax_mag.set_title("Espectro de Magnitud (centrado)")
                            ax_mag.set_xlabel("Frecuencia angular (rad/s)")
                            ax_mag.set_ylabel("Magnitud")
                            ax_mag.grid(True)
                            with col1:
                                st.pyplot(fig_mag)
                            fase = np.angle(fft_vals_shifted)
                            fig_fase, ax_fase = plt.subplots()
                            markerline, stemlines, baseline = ax_fase.stem(freqs_rad[filtro], fase[filtro], basefmt=" ")
                            plt.setp(markerline, color='orange', marker='o')
                            plt.setp(stemlines, color='orange')
                            ax_fase.set_title("Espectro de Fase (centrado y discreto)")
                            ax_fase.set_xlabel("Frecuencia angular (rad/s)")
                            ax_fase.set_ylabel("Fase (rad)")
                            ax_fase.grid(True)
                            with col2:
                                st.pyplot(fig_fase)
                            qol1,qol2=st.columns(2)
                            with qol1:
                                # --- Selección tipo de filtro ---
                                tipo_filtro = st.radio("Tipo de filtro:", ["Pasa bajos", "Pasa altos", "Pasa banda"])

                            
                                # --- Selección de frecuencias de corte ---
                                if tipo_filtro == "Pasa bajos":
                                    f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.0, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                    mascara = np.abs(freqs_rad) <= f_corte

                                elif tipo_filtro == "Pasa altos":
                                    f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.0, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                    mascara = np.abs(freqs_rad) >= f_corte

                                elif tipo_filtro == "Pasa banda":
                                    f1 = st.number_input("Frecuencia mínima (rad/s)", min_value=0.0, max_value=nyquist*2*np.pi, value=nyquist*np.pi/2, step=0.1)
                                    f2 = st.number_input("Frecuencia máxima (rad/s)", min_value=f1, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                    mascara = (np.abs(freqs_rad) >= f1) & (np.abs(freqs_rad) <= f2)

                            # --- Aplicar filtro al espectro ---
                            fft_filtrada_shifted = fft_vals_shifted * mascara
                            fft_filtrada = np.fft.ifftshift(fft_filtrada_shifted)  # volver al orden original
                            y_filtrada = np.fft.ifft(fft_filtrada)  # antitransformada
                            
                            # --- Magnitud del espectro filtrado vs original ---
                            with qol2:

                                fig_mag_comp, ax_mag_comp = plt.subplots()
                                ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_vals_shifted[filtro]), '--', label="Original", color='gray', alpha=0.5)
                                ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_filtrada_shifted[filtro]), label="Filtrada", color='blue')
                                ax_mag_comp.set_title("Magnitud del espectro: original vs filtrado")
                                ax_mag_comp.set_xlabel("Frecuencia angular (rad/s)")
                                ax_mag_comp.set_ylabel("Magnitud")
                                ax_mag_comp.grid(True)
                                ax_mag_comp.legend()
                                st.pyplot(fig_mag_comp)

                            # --- Visualizar reconstrucción ---
                            st.markdown("### 🔁 Señal filtrada (dominio del tiempo)")
                            rol1,rol2=st.columns(2)

                            mostrar_comparacion = st.checkbox("Comparar con señal original", value=True)
                            guardar_curva_filtrada = st.button("Guardar Curva Filtrada", key="GC")

                            if guardar_curva_filtrada:
                                st.session_state["CurvaFiltrada"] = np.real(y_filtrada[:M])

                            fig_filtrada, axf = plt.subplots()
                            if mostrar_comparacion:
                                if usar_curva_filtrada and st.session_state["CurvaFiltrada"] is not None:
                                    axf.plot(x_plot, st.session_state["CurvaFiltrada"], '--', label="Original", color='gray')
                                else:
                                    axf.plot(x_plot, y_plot, '--', label="Original", color='gray')
                            axf.plot(x_plot, np.real(y_filtrada[:M]), label="Filtrada", color='blue')
                            axf.set_xlabel(xlabel)
                            axf.set_ylabel(ylabel)
                            axf.set_title("Reconstrucción filtrada")
                            axf.grid(True)
                            axf.legend()
                            with rol1:
                                st.pyplot(fig_filtrada)

                            # --- Mostrar fase filtrada opcionalmente ---
                            fase_filtrada = np.angle(fft_filtrada_shifted)
                            fig_fase_filt, ax_fase_filt = plt.subplots()
                            markerline_filt, stemlines_filt, baseline_filt = ax_fase_filt.stem(freqs_rad[filtro], fase_filtrada[filtro], basefmt=" ")
                            plt.setp(markerline_filt, color='purple', marker='o')
                            plt.setp(stemlines_filt, color='purple')
                            ax_fase_filt.set_title("Espectro de Fase Filtrado (centrado y discreto)")
                            ax_fase_filt.set_xlabel("Frecuencia angular (rad/s)")
                            ax_fase_filt.set_ylabel("Fase (rad)")
                            ax_fase_filt.grid(True)
                            with rol2:
                                st.pyplot(fig_fase_filt)
# ------------------------------
# Mostrar resultado
# ------------------------------
# --------------------------------------------------------
### --------------------------------------------------
### --------------------------------------------------
    case "Reconstrucción tomográfica":
        #with st.expander("Módulo 4",expanded=True):
        st.title("💻 Módulo 3: Reconstrucción tomográfica")
        cool, _ = st.columns([1,5])
        with cool:
                if st.button("🔄 Reiniciar"):
                    st.session_state['I'] = None
                    st.session_state['I2'] = None
                    st.session_state["O"] = None
                    st.session_state["getp"] = None
                    st.session_state["geto"] = None
                    st.session_state["reconstrucciones"] = None
                    st.session_state["vid"] = None
                    st.session_state["fig1"] = None
                    st.session_state["fig2"] = None
                    st.session_state["fig3"] = None
                    st.session_state["Ani"] = None
                    st.session_state["arregloimg"] = None
                    st.session_state["loglikelihoods"] = None
                    archivo_I = None
                    archivo_I2 = None
                    vale3 = 1
        modo_sim = st.selectbox("Modo de adquisición",["TC", "SPECT"],key="modo_sim")
        modo_sim_bool = True if st.session_state["modo_sim"] == "SPECT" else False
        colum1, colum2 = st.columns(2)
        with colum1:
            # 2.1) Selección unificada
            atten_sel = st.selectbox(
                "Mapa de atenuación",
                ["Shepp-Logan", "Fantoma-Discos", "Fantoma-Tórax", "Fantoma-Agua", "Fantoma-Aire", "Subir mi imagen"],
                key="atten_sel"
            )

            # 2.2) Carga o fantoma + preview
            if atten_sel == "Subir mi imagen":
                archivo_I = st.file_uploader("Subí tu propio mapa de atenuación", type=["png","jpg","jpeg"])
                if archivo_I:
                    I = Image.open(archivo_I).convert("L")
                    I = np.array(I)
                    I = resize(I, (128,128), anti_aliasing=True, preserve_range=True)
                    I = I.astype(np.float32) / 255.0  # 👈 Acá la corrección

                    I_temp = I.copy()
                    I_limpia = I.copy()
                    st.image(I, caption="Vista previa", width=150, clamp=True)

                    if modo_sim == "TC":
                        st.subheader("🧪 Ruido en la imagen original")
                        add_noise = st.checkbox("Agregar ruido gaussiano",key="addnoise1")
                        if add_noise:
                            sigma = st.slider("Desvío estándar del ruido", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                            I_temp += np.random.normal(0, sigma * I_temp, I_temp.shape)
                            st.image(np.clip(I_temp, 0, 1), caption="Imagen con ruido", width=150, clamp=True)
                    #col_preview, _ = st.columns([1, 5])
                    #with col_preview:
                    
            else:
                if atten_sel == "Shepp-Logan":
                    I = 0.6 * shepp_logan_phantom()
                elif atten_sel == "Fantoma-Discos":
                    I = fn.fantoma_discos_atenuacion()
                elif atten_sel == "Fantoma-Tórax":
                    I = fn.fantoma_torax_atenuacion()
                elif atten_sel == "Fantoma-Agua":
                    I = fn.fantoma_agua_atenuacion()
                else:
                    I = fn.fantoma_aire()
                I = resize(I, (128,128), anti_aliasing=True, preserve_range=True)
                st.image(I, caption=atten_sel, width=150)
                I_temp = I.copy()
                I_limpia = I.copy()
                ###---------
                if modo_sim == "TC":
                    st.subheader("🧪 Ruido en la imagen original")
                    add_noise = st.checkbox("Agregar ruido gaussiano",key="addnoise2")
                    if add_noise:
                        sigma = st.slider("Desvío estándar del ruido", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                        I_temp += np.random.normal(0, sigma * I_temp, I_temp.shape)
                        st.image(np.clip(I_temp, 0, 1), caption="Imagen con ruido", width=150, clamp=True)
                    #col_preview, _ = st.columns([1, 5])

            # Ahora I está listo: usa I_temp = I.copy() si es necesario
            #I_temp = I.copy()
        with colum2:    
            if modo_sim == "SPECT":
                act_sel = st.selectbox(
                    "Mapa de actividad",
                    ["Fantoma de círculos", "Fantoma de díscos", "Subir mi imagen"],
                    key="act_sel"
                )
                if act_sel == "Subir mi imagen":
                    archivo_I2 = st.file_uploader("Subí tu mapa de actividad", type=["png","jpg","jpeg"])
                    if archivo_I2:
                        I2 = Image.open(archivo_I2).convert("L")
                        I2 = np.array(I2)
                        I2 = resize(I2, (128,128), anti_aliasing=True, preserve_range=True)
                        I2 = I2.astype(np.uint8)
                        #st.image(I2, caption="Tu mapa de actividad", width=150)
                        I_temp2 = I2.copy()
                        I_limpia2 = I2.copy()
                        #col_preview2, _ = st.columns([1, 5])
                        #with col_preview2:
                        st.image(I2, caption="Vista previa", width=150)
                        
                elif act_sel == "Fantoma de círculos":
                    I2 = fn.crear_mapa_actividad(N=128, radio=4, A1=100, desplazamiento=40)
                    st.image(I2/100, caption="Fantoma de círculos", width=150, clamp=True)
                    #I2 = I2.astype(np.uint8)
                    I_temp2 = I2.copy()
                    
                else:
                    I2 = fn.fantoma_discos_actividad()
                    st.image(I2/200, caption="Fantoma de díscos", width=150, clamp=True)
                    #I2 = I2.astype(float)
                    I_temp2 = I2.copy()
                # I_temp2 = I2.copy()
                I_limpia2 = I2.copy()
                st.subheader("🧪 Ruido en la imagen original")
                add_noise = st.checkbox("Agregar ruido gaussiano",key="addnoise3")
                if add_noise:
                    sigma_rel = st.slider("Desvío relativo del ruido", 0.0, 0.5, 0.05, 0.01)
                    ruido = np.random.normal(0, sigma_rel * I_temp2, I_temp2.shape)
                    I_temp2 += ruido
                    I_temp2 = np.clip(I_temp2, 0, None)  # Evita negativos
                    st.image(I_temp2 / np.max(I_temp2), caption="Con ruido", width=150, clamp=True)

        if "I2" not in st.session_state:
            st.session_state["I2"]=None
        if "I" not in st.session_state:
            st.session_state["I"]=None
        if "O" not in st.session_state:
            st.session_state["O"]=None
        if "getp" not in st.session_state:
            st.session_state["getp"]=None
        if "I" not in st.session_state:
            st.session_state["geto"]=None
        if "reconstrucciones" not in st.session_state:
            st.session_state["reconstrucciones"] = None
        if "fig1" not in st.session_state:
            st.session_state["fig1"] = None
        if "fig2" not in st.session_state:
            st.session_state["fig2"] = None
        if "fig3" not in st.session_state:
            st.session_state["fig3"] = None
        if "vid" not in st.session_state:
            st.session_state["vid"] = None
        if "arregloimg" not in st.session_state:
            st.session_state["arregloimg"] = None
        if "loglikelihoods" not in st.session_state:
            st.session_state["loglikelihoods"] = None
        if "I_limpia2" not in st.session_state:
            st.session_state["I_limpia2"] = None
        if "I_limpia" not in st.session_state:
            st.session_state["I_limpia"] = None
        col1, col2 = st.columns(2)
        with col1:
            st.header("⚙️ Método de recostrucción")
            operacion = st.radio("Elejí un método", ["FBP", "SART", "MLEM", "OSEM"], key="metodo_reconstruccion")
            operacion = st.session_state["metodo_reconstruccion"]
            modo_fov = st.radio("¿Cómo querés manejar la realación imagen-FOV?",
                                            ["Estándar",
                                            "Expandir imagen",
                                            "Eliminar fuera del FOV"],key="modo_fov")
            modo_fov = st.session_state["modo_fov"]
        with col2:     
            st.subheader("🌀 Parámetros de proyección")
            a = st.number_input("Ángulo inicial (°)", value=0, step=1)
            b = st.number_input("Ángulo final (°)", value=180, step=1)
            p = st.number_input("Paso (°)", value=1.0, step=0.1)#, min_value=1.0)
            if p <= 0:
                st.error("⚠️ El paso 'p' debe ser un número positivo.")
                st.stop()

            if a >= b:
                st.error("⚠️ El valor de 'a' debe ser menor que 'b'.")
                st.stop()
            if p >= (b - a):
                st.error("⚠️ El paso debe ser menor que la diferencia entre a y b.")
                st.stop()
        with col1:
            match modo_fov:
                case "Estándar":
                    st.write("El FOV tiene un radio igual a la altura de la matriz." \
                             " Si existen píxeles con valores distintos de cero fuera del FOV puede haber artefactos.")
                case "Expandir imagen":
                    st.write("Expande la imagen de forma rectangular con píxeles de valor cero." \
                             " La imagen original queda centrada en la imagen expandida y se sitúa dentro del FOV.")
                case "Eliminar fuera del FOV":
                    st.write("Lleva a cero los valores de la imagen fuera del FOV")
            #modo_vid=st.radio("Modo video",["Desctivado", "Activado"],key="modo_vid")
        #modo_vid_bool = True if st.session_state["modo_vid"] == "Activado" else False
        if operacion == "OSEM":
                    with col2:
                        #st.subheader("🔁 Iteraciones")
                        N = st.number_input("🔁 Iteraciones", min_value=1, value=10, step=1,format="%d")
                        s = st.number_input("Subsets", min_value=1, value=4, step=1,format="%d")
                        modo_O = st.radio("Elige tu imagen inicial:",["Imagen en blanco", "Imagen por FBP"])
                        #st.subheader("Subsets")
        if operacion == "MLEM":
                    with col2:
                        #st.subheader("🔁 Iteraciones")
                        N = st.number_input("🔁 Iteraciones", min_value=1, value=10, step=1,format="%d")
                        modo_O = st.radio("Elige tu imagen inicial:",["Imagen en blanco", "Imagen por FBP"])
        if operacion == "SART":
                    with col2:
                        #st.subheader("🔁 Iteraciones")
                        N = st.number_input("🔁 Iteraciones", min_value=1, value=10, step=1,format="%d")
        if atten_sel == "Subir mi imagen":
            if archivo_I is not None:
                match modo_fov:
                        case "Estándar":
                            pass
                        case "Expandir imagen":
                            I_temp = fn.expandir_a_fov(I_temp)
                            I_limpia = fn.expandir_a_fov(I_limpia)
                        case "Eliminar fuera del FOV":
                            I_temp = fn.aplicar_mascara_circular(I_temp,factor_radio=1)
                            I_limpia = fn.aplicar_mascara_circular(I_limpia,factor_radio=1)
            else:
                st.error("⚠️Debe cargar una imagen")
                st.stop()
        else:
             match modo_fov:
                        case "Estándar":
                            pass
                        case "Expandir imagen":
                            I_temp = fn.expandir_a_fov(I_temp)
                            I_limpia = fn.expandir_a_fov(I_limpia)
                        case "Eliminar fuera del FOV":
                            I_temp = fn.aplicar_mascara_circular(I_temp,factor_radio=1)
                            I_limpia = fn.aplicar_mascara_circular(I_limpia,factor_radio=1)
        if modo_sim == "SPECT":
                if act_sel == "Subir mi imagen":
                    if archivo_I2 is not None:
                        match modo_fov:
                            case "Estándar":
                                pass
                            case "Expandir imagen":
                                I_temp2 = fn.expandir_a_fov(I_temp2)
                                I_limpia2 = fn.expandir_a_fov(I_limpia2)
                            case "Eliminar fuera del FOV":
                                I_temp2 = fn.aplicar_mascara_circular(I_temp2,factor_radio=1)
                                I_limpia2 = fn.aplicar_mascara_circular(I_limpia2,factor_radio=1)
                    else:
                        st.error("⚠️Debe cargar una imagen")
                        st.stop()
                else:
                    match modo_fov:
                            case "Estándar":
                                pass
                            case "Expandir imagen":
                                I_temp2 = fn.expandir_a_fov(I_temp2)
                                I_limpia2 = fn.expandir_a_fov(I_limpia2)
                            case "Eliminar fuera del FOV":
                                I_temp2 = fn.aplicar_mascara_circular(I_temp2,factor_radio=1)
                                I_limpia2 = fn.aplicar_mascara_circular(I_limpia2,factor_radio=1)

        if modo_sim == "SPECT":
            angulos = np.arange(a, b, p)
            #assert I_temp.shape == I_temp2.shape
            sinograma = fn.forward_projection_simple(
                            activity=I_temp2,
                            mu=I_temp,
                            angulos=angulos,
                            circle=True
                            )
            sino = radon(I_temp2,angulos,circle=True)
                #st.session_state["getp"] = sinograma
            match operacion:
                case "FBP":
                    if st.button("Reconstruir"):
                        I,O,getp,geto,reconstrucciones = fn.FBP(I_temp,a,b,p,sinograma=sinograma)
                        st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"]=I_temp2,O,sino,geto
                        st.session_state["I_limpia2"] = I_limpia2
                case "MLEM":
                    if st.button("Reconstruir"):
                        I,O,getp,geto,arregloimg,loglikelihoods =fn.MLEM(I_temp, N, a, b, p,modo_O,I_limpia2,sinograma=sinograma)
                        st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"],st.session_state["arregloimg"],st.session_state["loglikelihoods"]=I_temp2,O,sino,geto,arregloimg,loglikelihoods
                        st.session_state["I_limpia2"] = I_limpia2
                case "OSEM":
                    if st.button("Reconstruir"):
                        I,O,getp,geto,arregloimg,loglikelihoods =fn.OSEM(I_temp, N, a, b, p,s,modo_O,I_limpia2,sinograma=sinograma)
                        st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"],st.session_state["arregloimg"],st.session_state["loglikelihoods"]=I_temp2,O,sino,geto,arregloimg,loglikelihoods
                        st.session_state["I_limpia2"] = I_limpia2
                case "SART":
                    if st.button("Reconstruir"):
                        I,O,getp,geto,arregloimg = fn.SART(I_temp, N, a, b, p,sinograma=sinograma)
                        st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"],st.session_state["arregloimg"]=I_temp2,O,sino,geto,arregloimg
                        st.session_state["I_limpia2"] = I_limpia2
            match operacion:
                case "FBP":
                    if st.button("Generar Gif",key="AF"):
                        st.session_state["Ani"] = None
                        st.session_state["vid"] = None
                        reconstrucciones = fn.FBP_vid(I_temp,a,b,p,sinograma=sinograma)
                        st.session_state["reconstrucciones"] = reconstrucciones
                        st.session_state["Ani"] = "AF"
                case "SART":
                    if st.button("Generar Gif",key="SA"):
                            st.session_state["Ani"] = None
                            st.session_state["vid"] = None
                            reconstrucciones = fn.SART_vid(I_temp,N,a,b,p,sinograma=sinograma)
                            st.session_state["reconstrucciones"] = reconstrucciones
                            st.session_state["Ani"] = "SA"
                case "MLEM":
                    if st.button("Generar Gif",key="AM"):
                            st.session_state["Ani"] = None
                            st.session_state["vid"] = None
                            reconstrucciones = fn.MLEM_vid(I_temp,N,a,b,p,modo_O,sinograma=sinograma)
                            st.session_state["reconstrucciones"] = reconstrucciones
                            st.session_state["Ani"] = "AM"
                case "OSEM":
                    if st.button("Generar Gif",key="OS"):
                        st.session_state["Ani"] = None
                        st.session_state["vid"] = None
                        reconstrucciones = fn.OSEM_vid(I_temp,N,a,b,p,s,modo_O,sinograma=sinograma)
                        st.session_state["reconstrucciones"] = reconstrucciones
                        st.session_state["Ani"] = "OS"
            
            yoli1,yoli2 = st.columns(2)
            with yoli1:
                ###---------------
                if st.session_state["arregloimg"] is not None and st.session_state["I_limpia2"] is not None  and st.session_state["I_limpia2"].shape == st.session_state["arregloimg"][0].shape:
                    if operacion != "FBP":
                        errores = fn.calcular_nrmse_series(st.session_state["arregloimg"], st.session_state["I_limpia2"])
                        # Encontrar el índice y valor mínimo
                        indice_min = np.argmin(errores)
                        valor_min = errores[indice_min]
                        fig_err, ax_err = plt.subplots()
                        ax_err.scatter(range(1,len(errores)+1), errores, color="blue")
                        ax_err.plot(indice_min + 1, valor_min, "ro", label="Mínimo")
                        ax_err.set_title("N-RMSE vs Iteraciones")
                        ax_err.set_xlabel("Iteración")
                        ax_err.set_ylabel("N-RMSE")
                        ax_err.grid(True)
                        st.pyplot(fig_err)
                        st.markdown(f"📉 **Error final (última iteración):** {errores[-1]:.4f}")
                        st.markdown(f"🔻 **Mínimo N-RMSE** en la iteración {indice_min + 1}: {valor_min:.4f}")
                    else:
                        error_final = fn.calcular_nrmse(st.session_state["O"],st.session_state["I_limpia2"])
                        st.markdown(f"📉 **N-RMSE reconstrucción final vs. original:** {error_final:.4f}")
            with yoli2:
                if st.session_state["loglikelihoods"] is not None and operacion != "FBP" and operacion != "SART" and st.session_state["I_limpia2"] is not None  and st.session_state["I_limpia2"].shape == st.session_state["arregloimg"][0].shape:
                    fig_ll, ax_ll = plt.subplots()
                    ax_ll.scatter(range(1,len(st.session_state["loglikelihoods"])+1),np.array(st.session_state["loglikelihoods"]), color="b")
                    ax_ll.set_title("Log-Likelihood vs Iteraciones")
                    ax_ll.set_xlabel("Iteración")
                    ax_ll.set_ylabel("Log-Likelihood")
                    ax_ll.grid(True)
                    st.pyplot(fig_ll)
                ###----------------------------------
            if st.session_state["I"] is not None and st.session_state["O"] is not None and st.session_state["getp"] is not None and st.session_state["geto"] is not None:
                
                verimagensinr = st.checkbox("Ver imagen orinal sin ruido",key="viosr")
                fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
                if verimagensinr:
                    im0 = axs1[0].imshow(st.session_state["I_limpia2"], cmap='gray')#,vmin=0, vmax=1)
                else:
                    im0 = axs1[0].imshow(st.session_state["I"], cmap='gray')#,vmin=0, vmax=1)
                axs1[0].set_title("Imagen original")
                axs1[0].axis("off")
                fig1.colorbar(im0, ax=axs1[0], fraction=0.046, pad=0.04)

                im1 = axs1[1].imshow(st.session_state["O"], cmap='gray')
                axs1[1].set_title("Imagen reconstruida")
                axs1[1].axis("off")
                fig1.colorbar(im1, ax=axs1[1], fraction=0.046, pad=0.04)
                #fig1.tight_layout()
                st.session_state["fig1"] = fig1
                st.session_state["fig1"].tight_layout()
                #st.pyplot(fig1)

                fig2, axs2 = plt.subplots(1, 2, figsize=(5, 10))
                axs2[0].imshow(fn.normalizar(st.session_state["getp"].T), cmap='gray')#,vmin=0, vmax=1)
                axs2[0].set_title("Sinograma original")
                axs2[0].axis("off")
                axs2[1].imshow(fn.normalizar(st.session_state["geto"].T), cmap='gray')#,vmin=0, vmax=1)
                axs2[1].set_title("Sinograma reconstriodo")
                axs2[1].axis("off")
                #fig2.tight_layout()
                st.session_state["fig2"] = fig2
                st.session_state["fig2"].tight_layout()

            if st.session_state["fig1"] is not None and st.session_state["fig2"] is not None:
                st.subheader("🖼️ Imágenes")
                st.pyplot(st.session_state["fig1"])
                st.subheader("📈 Sinogramas")
                st.pyplot(st.session_state["fig2"])
            if st.session_state["reconstrucciones"] is not None and st.session_state["Ani"] is not None and st.session_state["vid"] is None:
                    #st.subheader("🎞️ Animación de la reconstrucción")
                    #paso = st.slider("Paso", 1, len(st.session_state["reconstrucciones"]), 1)
                    if "video_generado" not in st.session_state:
                        with st.spinner("🎥 Generando animación..."):
            # Primero el GIF para previsualizar con st.image
                            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_gif:
                                fn.crear_video_reconstruccion(
                                    st.session_state["reconstrucciones"],
                                    output_path=temp_gif.name,
                                    writer="pillow"
                                    )
                                st.session_state["vid"] = temp_gif.name
                                #st.image(temp_gif.name)
            if st.session_state["vid"] is not None and st.session_state["Ani"] is not None:
                st.subheader("🎞️ Animación de la reconstrucción")
                #with st.spinner("🎥 Generando animación..."):
                st.image(st.session_state["vid"])

        else:
            match operacion:
                case "FBP":
                        if st.button("Reconstruir"):
                                I,O,getp,geto,reconstrucciones = fn.FBP(I_temp, a, b, p)
                                #if modo_vid_bool:
                                #    st.session_state["reconstrucciones"] = reconstrucciones
                                st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"]=I,O,getp,geto
                                st.session_state["I_limpia"] = I_limpia
                case "MLEM":
                            if st.button("Reconstruir"):
                                I,O,getp,geto,arregloimg,loglikelihoods =fn.MLEM(I_temp, N, a, b, p,modo_O,I_limpia)
                                st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"],st.session_state["arregloimg"],st.session_state["loglikelihoods"]=I,O,getp,geto,arregloimg,loglikelihoods
                                st.session_state["I_limpia"] = I_limpia
                case "OSEM":
                        if st.button("Reconstruir"):
                            I,O,getp,geto,arregloimg,loglikelihoods =fn.OSEM(I_temp, N, a, b, p,s,modo_O,I_limpia)
                            st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"],st.session_state["arregloimg"],st.session_state["loglikelihoods"]=I,O,getp,geto,arregloimg,loglikelihoods
                            st.session_state["I_limpia"] = I_limpia
                case "SART":
                        if st.button("Reconstruir"):
                            I,O,getp,geto,arregloimg = fn.SART(I_temp, N, a, b, p)
                            st.session_state["I"],st.session_state["O"],st.session_state["getp"],st.session_state["geto"],st.session_state["arregloimg"]=I,O,getp,geto,arregloimg
                            st.session_state["I_limpia"] = I_limpia
            match operacion:
                case "FBP":
                    if st.button("Generar Gif",key="AF"):
                        st.session_state["Ani"] = None
                        st.session_state["vid"] = None
                        reconstrucciones = fn.FBP_vid(I_temp,a,b,p,)
                        st.session_state["reconstrucciones"] = reconstrucciones
                        st.session_state["Ani"] = "AF"
                case "SART":
                    if st.button("Generar Gif",key="SA"):
                            st.session_state["Ani"] = None
                            st.session_state["vid"] = None
                            reconstrucciones = fn.SART_vid(I_temp,N,a,b,p)
                            st.session_state["reconstrucciones"] = reconstrucciones
                            st.session_state["Ani"] = "SA"
                case "MLEM":
                    if st.button("Generar Gif",key="AM"):
                            st.session_state["Ani"] = None
                            st.session_state["vid"] = None
                            reconstrucciones = fn.MLEM_vid(I_temp,N,a,b,p,modo_O)
                            st.session_state["reconstrucciones"] = reconstrucciones
                            st.session_state["Ani"] = "AM"
                case "OSEM":
                    if st.button("Generar Gif",key="OS"):
                        st.session_state["Ani"] = None
                        st.session_state["vid"] = None
                        reconstrucciones = fn.OSEM_vid(I_temp,N,a,b,p,s,modo_O)
                        st.session_state["reconstrucciones"] = reconstrucciones
                        st.session_state["Ani"] = "OS"
                
            yoli1,yoli2 = st.columns(2)
            with yoli1:
                ###---------------
                if st.session_state["arregloimg"] is not None and st.session_state["I_limpia"] is not None  and st.session_state["I_limpia"].shape == st.session_state["arregloimg"][0].shape:
                    if operacion != "FBP":
                        errores = fn.calcular_nrmse_series(st.session_state["arregloimg"], st.session_state["I_limpia"])
                        # Encontrar el índice y valor mínimo
                        indice_min = np.argmin(errores)
                        valor_min = errores[indice_min]
                        fig_err, ax_err = plt.subplots()
                        ax_err.scatter(range(1,len(errores)+1), errores, color="blue")
                        ax_err.plot(indice_min + 1, valor_min, "ro", label="Mínimo")
                        ax_err.set_title("N-RMSE vs Iteraciones")
                        ax_err.set_xlabel("Iteración")
                        ax_err.set_ylabel("N-RMSE")
                        ax_err.grid(True)
                        st.pyplot(fig_err)
                        st.markdown(f"📉 **Error final (última iteración):** {errores[-1]:.4f}")
                        st.markdown(f"🔻 **Mínimo N-RMSE** en la iteración {indice_min + 1}: {valor_min:.4f}")
                    else:
                        error_final = fn.calcular_nrmse(st.session_state["O"],st.session_state["I_limpia"])
                        st.markdown(f"📉 **N-RMSE reconstrucción final vs. original:** {error_final:.4f}")
            with yoli2:
                if st.session_state["loglikelihoods"] is not None and st.session_state["I_limpia"] is not None and operacion != "FBP" and operacion != "SART" and st.session_state["I_limpia"].shape == st.session_state["arregloimg"][0].shape:
                    fig_ll, ax_ll = plt.subplots()
                    ax_ll.scatter(range(1,len(st.session_state["loglikelihoods"])+1),np.array(st.session_state["loglikelihoods"]), color="b")
                    ax_ll.set_title("Log-Likelihood vs Iteraciones")
                    ax_ll.set_xlabel("Iteración")
                    ax_ll.set_ylabel("Log-Likelihood")
                    ax_ll.grid(True)
                    st.pyplot(fig_ll)
                ###----------------------------------
            if st.session_state["I"] is not None and st.session_state["O"] is not None and st.session_state["getp"] is not None and st.session_state["geto"] is not None:
                
                fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
                verimagensinrtc = st.checkbox("Ver imagen orinal sin ruido",key="viosrtc")
                if verimagensinrtc:
                    im0 = axs1[0].imshow(st.session_state["I_limpia"], cmap='gray')#,vmin=0, vmax=1)
                else:    
                    im0 = axs1[0].imshow(st.session_state["I"], cmap='gray')#,vmin=0, vmax=1)
                axs1[0].set_title("Imagen original")
                axs1[0].axis("off")
                fig1.colorbar(im0, ax=axs1[0], fraction=0.046, pad=0.04)

                im1 = axs1[1].imshow(st.session_state["O"], cmap='gray')
                axs1[1].set_title("Imagen reconstruida")
                axs1[1].axis("off")
                fig1.colorbar(im1, ax=axs1[1], fraction=0.046, pad=0.04)
                #fig1.tight_layout()
                st.session_state["fig1"] = fig1
                st.session_state["fig1"].tight_layout()
                #st.pyplot(fig1)

                fig2, axs2 = plt.subplots(1, 2, figsize=(5, 10))
                axs2[0].imshow(fn.normalizar(st.session_state["getp"].T), cmap='gray')#,vmin=0, vmax=1)
                axs2[0].set_title("Sinograma original")
                axs2[0].axis("off")
                axs2[1].imshow(fn.normalizar(st.session_state["geto"].T), cmap='gray')#,vmin=0, vmax=1)
                axs2[1].set_title("Sinograma reconstruido")
                axs2[1].axis("off")
                #fig2.tight_layout()
                st.session_state["fig2"] = fig2
                st.session_state["fig2"].tight_layout()
                #st.pyplot(fig2)
                if st.session_state["fig1"] is not None and st.session_state["fig2"] is not None:
                    st.subheader("🖼️ Imágenes")
                    st.pyplot(st.session_state["fig1"])
                    st.subheader("📈 Sinogramas")
                    st.pyplot(st.session_state["fig2"])

                colif1,colif2 = st.columns(2)
                with colif1:
                    window_center = st.slider('Centro de ventana (nivel, L)', min_value=-1000, max_value=1000, value=0, step=10)
                    window_width = st.slider('Ancho de ventana (W)', min_value=1, max_value=2000, value=400, step=10)
                    verimg = st.radio("Elegí que imagen ver en HU", ["Imagen original sin ruido", "Imagen original con ruido", "Imagen recosntruida"])
                    if verimg == "Imagen original sin ruido":
                        IUH = fn.convertir_a_hounsfield(st.session_state["I_limpia"])
                        cadena = "Imagen original s/r con ventaneo"
                    elif verimg =="Imagen original con ruido":
                        IUH = fn.convertir_a_hounsfield(st.session_state["I"])
                        cadena = "Imagen original c/r con ventaneo"
                    else:
                        IUH = fn.convertir_a_hounsfield(st.session_state["O"])
                        cadena = "Imagen reconstruida con ventaneo"
    # Aplicar ventaneo a la imagen
                L = window_center
                W = window_width
                I_ventaneado = np.clip((IUH - (L - W / 2)) / W, 0, 1)
                fig3, axs3 = plt.subplots(1,1,figsize=(5,10))
                im3 = axs3.imshow(I_ventaneado, cmap='gray')
                axs3.set_title(f'{cadena} ({L}, {W})')
                axs3.axis('off')
                im3 = fig3.colorbar(im3,ax= axs3,fraction=0.046, pad=0.04)
                st.session_state["fig3"] = fig3#plt.show()

            if st.session_state["fig3"] is not None:
                with colif2:
                    st.pyplot(st.session_state["fig3"])
            if st.session_state["reconstrucciones"] is not None and st.session_state["Ani"] is not None and st.session_state["vid"] is None:
                    #st.subheader("🎞️ Animación de la reconstrucción")
                    #paso = st.slider("Paso", 1, len(st.session_state["reconstrucciones"]), 1)
                    if "video_generado" not in st.session_state:
                        with st.spinner("🎥 Generando animación..."):
            # Primero el GIF para previsualizar con st.image
                            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_gif:
                                fn.crear_video_reconstruccion(
                                    st.session_state["reconstrucciones"],
                                    output_path=temp_gif.name,
                                    writer="pillow"
                                    )
                                st.session_state["vid"] = temp_gif.name
                                #st.image(temp_gif.name)
            if st.session_state["vid"] is not None and st.session_state["Ani"] is not None:
                st.subheader("🎞️ Animación de la reconstrucción")
                #with st.spinner("🎥 Generando animación..."):
                st.image(st.session_state["vid"])
        ###-------------------
    ##-----------------------------
    ##-----------------------------

    case "Procesamiento de imágenes":
        st.title("🖼️ Módulo 2: Procesamiento de imágenes")
        #if "archivoP" not in st.session_state:
        #    st.session_state["archivoP"] = None
        #if "archivoM" not in st.session_state:
        #    st.session_state["archivoM"] = None
        archivoP = None
        archivoM = None
        proy = None
        tabsp = st.tabs(["Cargador de imágenes", "Visualizador de imágenes", "Editor de imágenes", "Calibrador", "Analizador de imágenes"])
        with tabsp[0]:
            st.subheader("Cargadorde imágenes")
            subirarchivo = st.radio("Modo de subir archivo", ["Automático", "Manual (solo DICOM)"],horizontal=True )
            if subirarchivo == "Automático":
                archivoP = st.file_uploader("📂 Subí una imagen o archivo DICOM", 
                                            type=["dcm", "jpg", "jpeg", "png", "tiff", "bmp"])
                
                #if archivoP is not None:
                #    st.session_state["archivoP"] = archivoP
                #archivoP = st.session_state["archivoP"]
            
            else:
                # Entradas
                M = st.number_input('Número de Columnas (M):', min_value=1, step=1)
                N = st.number_input('Número de Filas (N):', min_value=1, step=1)
                I = st.number_input('Cantidad de Imágenes (I):', min_value=1, step=1)
                modo = st.selectbox('Modo:', ['1: Modo Byte', '2: Modo Word'])
                modo_val = 1 if modo.startswith('1') else 2

                # Carga de archivo
                archivoM = st.file_uploader("Subí un archivo binario")#, type=['bin', 'raw', 'dat', '*', "*.*"])

                #if archivoM is not None:
                #    st.session_state["archivoM"] = archivoM
                if archivoM is not None:
                    # archivoP es un "archivo virtual" tipo BytesIO
                    contenidoM = archivoM.read()
                    total_bytes_esperados = M * N * I * modo_val

                    if len(contenidoM) < total_bytes_esperados:
                        st.error(f"El archivo es demasiado pequeño. Esperado: {total_bytes_esperados} bytes, recibido: {len(contenidoM)}.")
                    else:
                        contenido_final = contenidoM[-total_bytes_esperados:]
                        tipo_dato = np.uint8 if modo_val == 1 else np.uint16
                        G = np.frombuffer(contenido_final, dtype=tipo_dato)

                        if G.size != M * N * I:
                            st.error("Cantidad de datos incorrecta tras convertir.")
                        else:
                            L = G.reshape((I, N, M))
                            proy = np.transpose(L, (1, 2, 0))  # (filas, columnas, imágenes)

                            st.success("Archivo procesado correctamente.")
            
            #for canal in ["canalR", "canalG", "canalB"]:
            #    if canal in st.session_state:
            #        del st.session_state[canal]
            for canal in ["canalR", "canalG", "canalB", "img_original"]:
                if canal not in st.session_state:
                    st.session_state[canal] = None
            #for fil in ["F_r", "F_g", "F_b", "Ffilt_r", "Ffilt_g", "Ffilt_b"]:
            #    if fil in st.session_state:
            #        del st.session_state[fil]
            img_t = None
            img_v = None

        #VIS=st.expander("Visualizador",expanded=False)
        with tabsp[1]:
            st.subheader("Visualizador de imágenes")
            if archivoP is not None:
                contenido = archivoP.read()
                try:
                    # Intentamos leer como DICOM
                    ds = pydicom.dcmread(io.BytesIO(contenido))
                    st.success("✅ Archivo DICOM detectado.")
                    
                    st.subheader("📋 Información DICOM:")
                    st.write({
                        "Paciente": ds.get("PatientName", "Desconocido"),
                        "Estudio": ds.get("StudyDescription", "N/A"),
                        "Modalidad": ds.get("Modality", "N/A"),
                        "Dimensiones": f"{ds.Rows} × {ds.Columns}",
                        "Photometric Interpretation": ds.get("PhotometricInterpretation", "N/A"),
                        "Frames": ds.get("NumberOfFrames", 1)
                    })

                    # Convertimos a imagen (solo el primer frame por ahora)
                    pixel_array = ds.pixel_array
                    st.session_state["proy"] = pixel_array
                    is_multiframe = hasattr(ds, "NumberOfFrames") and ds.NumberOfFrames > 1
                    col1, col2 = st.columns([0.2,0.8])

                    if is_multiframe:
                        st.write(f"Número de frames: {ds.NumberOfFrames}")
                        
                        with col1:
                            # Slider para elegir el frame
                            idx = st.slider("Elegí el frame", 0, ds.NumberOfFrames - 1, 0,key="Elegí el frame")
                        frame = pixel_array[idx]
                    else:
                        frame = pixel_array

                    
                    # Normalizar para mostrar
                    img = frame.astype(np.float32)
                    img_t = img.copy()
                    img -= img.min()
                    img /= img.max()

                    if is_multiframe:
                        #img = frame.astype(np.float32)
                        #img -= img.min()
                        #img /= img.max()
                        figuras, axs = plt.subplots(1,1,figsize=(5,10))
                        axs.imshow(img, cmap='gray')
                        #axs.set_title(f'{cadena} ({L}, {W})')
                        axs.axis('off')
                        with col2:
                            st.pyplot(figuras)
                        
                        #if "UV" not in st.session_state:
                        #    st.session_state["UV"] = None
                        #if st.button("Usar ventaneo",key="uv"):
                        #    st.session_state["UV"] = st.session_state["uv"]
                        #if st.session_state["UV"] is not None:
                        st.subheader("⚙️ Ajuste de ventana")
                        kol1, kol2 = st.columns([0.2,0.8])
                        with kol1:
                            idx2 = st.slider("Elegí el frame", 0, ds.NumberOfFrames - 1, 0,key="Elegí el frame 2")
                        frame2 = pixel_array[idx2]
                        img_original = frame2.astype(np.float32)
                        # Sacamos información para los sliders
                        imin = np.min(img_original)
                        imax = np.max(img_original)

                    #if st.session_state["UV"] is not None:
                        
                        # Intentamos extraer valores de ventaneo del DICOM
                        try:
                            window_center_default = float(ds.WindowCenter)
                            window_width_default = float(ds.WindowWidth)
                            if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
                                window_center_default = float(ds.WindowCenter[0])
                            if isinstance(ds.WindowWidth, pydicom.multival.MultiValue):
                                window_width_default = float(ds.WindowWidth[0])
                        except AttributeError:
                            # Si no existen, calculamos por defecto
                            window_center_default = (imin + imax) / 2
                            window_width_default = (imax - imin) / 2

                        with kol1:
                            window_center = st.slider('Centro de ventana (Nivel, L)', 
                                            min_value=float(imin), 
                                            max_value=float(imax), 
                                            value=window_center_default)

                            window_width = st.slider('Ancho de ventana (W)', 
                                            min_value=1.0, 
                                            max_value=float(imax-imin), 
                                            value=window_width_default)
                        # Aplicar ventaneo
                        ventana_min = window_center - (window_width/2)
                        ventana_max = window_center + (window_width/2)

                        img_v = np.clip((img_original - ventana_min) / (ventana_max - ventana_min), 0, 1)
                        # Mostrar imagen ventaneada con opción de negativo
                        mostrar_negativo = st.checkbox("Ver negativo", key="negativo_multi")
                        if mostrar_negativo:
                            img_v = 1.0 - img_v
                    # Mostrar imagen ventaneada
                        figura, ax = plt.subplots(figsize=(5, 10))
                        ax.imshow(img_v, cmap='gray')
                        ax.axis('off')
                    #if st.session_state["UV"] is not None:
                        with kol2:
                            #if VIS:
                                st.pyplot(figura)
                            #else:
                            #    pass
                        #else:
                        #    pass
                        
                        if "gifdicom" not in st.session_state:
                            st.session_state["gifdicom"] = None
                        if st.button("Generar Gif",key="GG"):
                            st.session_state["gifdicom"]=st.session_state["GG"]
                        if st.session_state["gifdicom"] is not None:
            #                if is_multiframe:
                            gif_path = fn.crear_gif_desde_multiframe(pixel_array)
                            #if VIS:
                            st.image(gif_path, caption="GIF animado del DICOM")#, use_container_width=True)
                            #else:
                            #    pass
                    else:
                        figuras, axs = plt.subplots(figsize=(5,10))
                        axs.imshow(img, cmap='gray')
                        #axs.set_title(f'{cadena} ({L}, {W})')
                        axs.axis('off')
                        with col2:
                            #if VIS:
                            st.pyplot(figuras)
                            #else:
                            #    pass
                        #if "UV" not in st.session_state:
                        #    st.session_state["UV"] = None
                        #if st.button("Usar ventaneo", key="uv_single"):
                        #    st.session_state["UV"] = st.session_state["uv_single"]
                        #if st.session_state["UV"] is not None:
                        st.subheader("⚙️ Ajuste de ventana")

                        img_original = frame.astype(np.float32)
                        imin = np.min(img_original)
                        imax = np.max(img_original)

                        # Intentar leer ventaneo del DICOM
                        try:
                            window_center_default = float(ds.WindowCenter)
                            window_width_default = float(ds.WindowWidth)
                            if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
                                window_center_default = float(ds.WindowCenter[0])
                            if isinstance(ds.WindowWidth, pydicom.multival.MultiValue):
                                window_width_default = float(ds.WindowWidth[0])
                        except AttributeError:
                            window_center_default = (imin + imax) / 2
                            window_width_default = (imax - imin) / 2

                        kol1, kol2 = st.columns([0.2, 0.8])
                        with kol1:
                            window_center = st.slider('Centro de ventana (Nivel, L)', 
                                                    min_value=float(imin), 
                                                    max_value=float(imax), 
                                                    value=window_center_default)

                            window_width = st.slider('Ancho de ventana (W)', 
                                                    min_value=1.0, 
                                                    max_value=float(imax-imin), 
                                                    value=window_width_default)
                            # Aplicar ventaneo
                            ventana_min = window_center - (window_width/2)
                            ventana_max = window_center + (window_width/2)
                            img_v = np.clip((img_original - ventana_min) / (ventana_max - ventana_min), 0, 1)
                            with kol1:
                                # Checkbox para invertir imagen
                                invertir = st.checkbox("Mostrar negativo", key="neg_single")
                                # Aplicar inversión si se selecciona
                                if invertir:
                                    img_v = 1.0 - img_v

                            # Mostrar imagen ventaneada (invertida o no)
                            figura, ax = plt.subplots(figsize=(5, 10))
                            ax.imshow(img_v, cmap='gray')
                            ax.axis("off")
                            with kol2:
                                #if VIS:
                                st.pyplot(figura)
                                #else:
                                #    pass
                except Exception as e:
                    # No es un DICOM → tratamos como imagen común
                    try:
                        img = Image.open(io.BytesIO(contenido))
                        img_t = img.copy()
                        st.success("✅ Imagen común detectada.")
                        st.image(img, caption="Imagen cargada", use_container_width=True)

                        st.subheader("📋 Información de la imagen:")
                        st.write({
                            "Formato": img.format,
                            "Tamaño": img.size,
                            "Modo": img.mode
                        })
                        
                        I_g = img.convert("L")
                        I_g = np.array(I_g)
                        imax = np.max(I_g)
                        imin = np.min(I_g)
                        if st.checkbox("Mostrar negativo", key="neg_comun"):
                            I_g = 1.0 - I_g
                        #I_g = fn.convertir_a_hounsfield(I_g)
                        window_center = st.slider('Centro de ventana (nivel, L)', min_value=imin,
                                                max_value=imax, value=int((imax - imin)/2), step=int(0.01*imax))
                        window_width = st.slider('Ancho de ventana (W)', min_value=1, 
                                                max_value=imax-imin, value=int((imax-imin)/10), step=int(0.01*imax))
                        L = window_center
                        W = window_width
                        img_v = np.clip((I_g - (L - W / 2)) / W, 0, 1)
                        figven, axs3 = plt.subplots(1,1,figsize=(5,10))
                        axs3.imshow(img_v, cmap='gray')
                        #axs3.set_title(f'{cadena} ({L}, {W})')
                        axs3.axis('off')
                        if "figven" not in st.session_state:
                            st.session_state["figven"] = None
                        st.session_state["figven"] = figven#plt.show()
                        #st.pyplot(fig2)
                        if st.session_state["figven"] is not None:
                            #if VIS:
                            st.pyplot(st.session_state["figven"])
                            #else:
                            #    pass
                        else:
                            pass
                    except:
                        st.error("❌ No se pudo leer el archivo como imagen o DICOM.")
            elif proy is not None and archivoM is not None:
                #st.subheader("📋 Información DICOM:")
                img0 = proy[:, :, :].astype(np.float32)
                for k in range(I-1):
                    imin = img0[:,:,k].min()
                    imax = img0[:,:,k].max()
                    img0[:,:,k] -= imin
                    img0[:,:,k] /= imax
                
                manualcol1, manualcol2 = st.columns([0.2,0.8])
                if I>1:
                    with manualcol1:
                        nimg = st.slider("Número de imagen", min_value=0,
                                                        max_value=I-1, value=0, step=1,key="nimg")
                        mostrar_negativo_manual = st.checkbox("Ver negativo", key="negativo_manual")
                        if mostrar_negativo_manual:
                            img0 = 1.0 - img0
                    with manualcol2:
                        figmanual, axmanual = plt.subplots()
                        axmanual.imshow(img0[:, :, nimg], cmap='gray')
                        axmanual.axis("off")
                        #if VIS:
                        st.pyplot(figmanual)
                        #else:
                        #    pass
                    mc1, mc2 = st.columns([0.2,0.8])
                    with mc1:
                        # Ventaneo
                        st.subheader("Ventaneo manual")
                        all_min = float(img0.min())
                        all_max = float(img0.max())
                        wc_default = (all_max + all_min) / 2
                        ww_default = (all_max - all_min)
                        
                        nimg2 = st.slider("Número de imagen", min_value=0,
                                                        max_value=I-1, value=0, step=1,key="nimg2")
                        wc = st.slider("Centro de ventana (Nivel - WC)", all_min, all_max, wc_default,key="wc2")
                        ww = st.slider("Ancho de ventana (WW)", 1e-5, all_max - all_min, ww_default,key="ww2")

                        vmin = wc - ww / 2
                        vmax = wc + ww / 2
                    with mc2:
                        img_v = np.clip((img0[:, :, nimg2] - vmin) / (vmax - vmin), 0, 1)
                    with mc1:
                        mostrar_negativo_manual_2 = st.checkbox("Ver negativo", key="negativo_manual_2")
                        if mostrar_negativo_manual_2:
                            img_v = 1.0 - img_v
                    with mc2:
                        figmanual2, axmanual2 = plt.subplots()
                        axmanual2.imshow(img_v, cmap='gray')
                        axmanual2.axis("off")
                        #if VIS:
                        st.pyplot(figmanual2)
                        
                    img_t = img0[:,:,nimg]
                    if "gifdicoma" not in st.session_state:
                        st.session_state["gifdicoma"] = None
                    if st.button("Generar Gif",key="GGa"):
                        st.session_state["gifdicoma"]=st.session_state["GGa"]
                    if st.session_state["gifdicoma"] is not None:
                #             if is_multiframe:
                        gif_path = fn.crear_gif_manual(proy,I,nombre="dicom_animado2.gif", fps=5)
                        #if VIS:
                        st.image(gif_path, caption="GIF animado del DICOM")
                        #else:
                        #    pass
                else:
                    #imax = np.max(proy[:,:,0])
                    #imin = np.min(proy[:,:,0])
                    colmanu1, colmanu2 =st.columns([0.2,0.8])
                    with colmanu2:
                        img_t = proy[:,:,0].astype(np.float32)
                        figmanual2, axmanual2 = plt.subplots()
                        axmanual2.imshow(proy, cmap='gray')
                        axmanual2.axis("off")
                        st.pyplot(figmanual2)
                    colman1, colman2 = st.columns([0.2,0.8])
                    #I_g = fn.convertir_a_hounsfield(I_g)
                    with colman1:
                        window_center = st.slider('Centro de ventana (nivel, L)', min_value=-1000,
                                                max_value=3000, value=1000, step=10)
                        window_width = st.slider('Ancho de ventana (W)', min_value=1, 
                                                max_value=4000, value=1000, step=10)
                        L = window_center
                        W = window_width
                        img_v = np.clip((img0[:,:,0] - (L - W / 2)) / W, 0, 1)
                        mostrar_negativo_manual_3 = st.checkbox("Ver negativo", key="negativo_manual_3")
                        if mostrar_negativo_manual_3:
                            img_v = 1.0 - img_v
                    figmanual, axmanual = plt.subplots()
                    axmanual.imshow(img_v, cmap='gray')
                    axmanual.axis("off")
                    with colman2:
                        #if VIS:
                        st.pyplot(figmanual)
                        #else:
                        #    pass
        #EDI=st.expander("Editor de imágenes",expanded=False)
        with tabsp[2]:
            st.subheader("Editor de imágenes")
            img_a = None
            if "img_gris" not in st.session_state:
                st.session_state["img_gris"] = None
            if img_t is not None:
                CEG0 = st.radio("¿Qué imagen deseas editar?", ["Imagen Original", "Imagen Ventaneada"],horizontal=True)
                if CEG0 == "Imagen Original":
                    if img_t is not None:
                        img_a = np.array(img_t)
                else:
                    if img_v is not None:
                        img_a = np.array(img_v)
            if img_a is not None:
                #img_a = np.array(img_t)

                if img_a.ndim == 3 and img_a.shape[2] in [3,4]:  # Imagen RGB
                    CEG = st.radio("Convertir a escala de grisis", ["No", "Sí"], horizontal=True)
                    if CEG == "Sí":
                        img_a = img.convert("L")
                        img_a = np.array(img_a)
                    else:
                        pass
                if img_a.ndim == 3 and img_a.shape[2] in [3,4]:  # Imagen RGB
                    # Comparar con la imagen anterior
                    if "img_original" not in st.session_state or not np.array_equal(img_a, st.session_state["img_original"]):
                        # 🔄 Se subió una nueva imagen → reiniciar todos los canales y operaciones
                        st.session_state["img_original"] = img_a
                        st.session_state["canalR"] = img_a[:, :, 0]
                        st.session_state["canalG"] = img_a[:, :, 1]
                        st.session_state["canalB"] = img_a[:, :, 2]
                        # Opcional: reiniciar otras cosas si las usás
                        st.session_state["F_r"] = None
                        st.session_state["F_g"] = None
                        st.session_state["F_b"] = None
                        st.session_state["Ffilt_r"] = None
                        st.session_state["Ffilt_g"] = None
                        st.session_state["Ffilt_b"] = None
                        st.session_state["AplicarFR"] = None
                        st.session_state["AplicarFG"] = None
                        st.session_state["AplicarFB"] = None
                        st.session_state["recombinar"] = None
                        # Agregá más si tenés otras operaciones
                    # Verificamos si la imagen cambió comparando con una guardada
                    if "img_original" not in st.session_state or not np.array_equal(img_a, st.session_state["img_original"]):
                        # Guardamos la imagen completa para referencia
                        st.session_state["img_original"] = img_a
                        # Separar los canales
                        st.session_state["canalR"] = img_a[:, :, 0]
                        st.session_state["canalG"] = img_a[:, :, 1]
                        st.session_state["canalB"] = img_a[:, :, 2]
                    OPERACION = st.selectbox("Elegí una operación:", ["Rotar", "Contraste y brillo", "Operaciones morfológicas",
                                                                    "Filtro espacial", "Filtro frecuencial",
                                                                    "Ecualización"])
                    match OPERACION:
                        case "Rotar":
                            st.header("🎛️ Rotaciones Independientes de Canales")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.subheader("🔴 Canal Rojo")
                                angulo_R = st.slider("Ángulo de rotación R", min_value=-180, max_value=180, value=0, step=1)
                                previa_Ifilt_r = scipy.ndimage.rotate(st.session_state["canalR"], angle=angulo_R, reshape=False, mode='nearest')
                                #figPrevR, axPrevR = plt.subplots(figsize=(7, 5))
                                #axPrevR.imshow(previa_Ifilt_r, cmap='gray')
                                #axPrevR.axis("off")
                                st.write("Img Rojo (Previsualización)")
                                #st.pyplot(figPrevR)
                                #fn.mostrar_figura_como_imagen(figPrevR)
                                #plt.close(figPrevR)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_r),use_container_width=True)
                                #else:
                                #    pass
                                if st.button("Rotar Rojo"):
                                    st.session_state["canalR"] = scipy.ndimage.rotate(st.session_state["canalR"], angle=angulo_R, reshape=False, mode='nearest')
                            with c2:
                                st.subheader("🟢 Canal Verde")
                                angulo_G = st.slider("Ángulo de rotación G", min_value=-180, max_value=180, value=0, step=1)
                                previa_Ifilt_g = scipy.ndimage.rotate(st.session_state["canalG"], angle=angulo_G, reshape=False, mode='nearest')
                                #figPrevG, axPrevG = plt.subplots(figsize=(7, 5))
                                #axPrevG.imshow(previa_Ifilt_g, cmap='gray')
                                #axPrevG.axis("off")
                                st.write("Img Verde (Previsualización)")
                                #st.pyplot(figPrevG)
                                #fn.mostrar_figura_como_imagen(figPrevG)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)
                                #else:
                                #    pass
                                #plt.close(figPrevG)
                                if st.button("Rotar Verde"):
                                    st.session_state["canalG"] = scipy.ndimage.rotate(st.session_state["canalG"], angle=angulo_G, reshape=False, mode='nearest')
                            with c3:
                                st.subheader("🔵 Canal Azul")
                                angulo_B = st.slider("Ángulo de rotación B", min_value=-180, max_value=180, value=0, step=1)
                                previa_Ifilt_b = scipy.ndimage.rotate(st.session_state["canalB"], angle=angulo_B, reshape=False, mode='nearest')
                                #figPrevB, axPrevB = plt.subplots(figsize=(7, 5))
                                #axPrevB.imshow(previa_Ifilt_b, cmap='gray')
                                #axPrevB.axis("off")
                                st.write("Img Azul (Previsualización)")
                                #st.pyplot(figPrevB)
                                #fn.mostrar_figura_como_imagen(figPrevB)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)
                                #else:
                                #    pass
                                #plt.close(figPrevB)
                                if st.button("Rotar Azul"):
                                    st.session_state["canalB"] = scipy.ndimage.rotate(st.session_state["canalB"], angle=angulo_B, reshape=False, mode='nearest')
                        case "Contraste y brillo":
                            st.header("✨ Ajustes de Brillo y Contraste por Canal")
                            # Crear columnas para cada canal
                            c1, c2, c3 = st.columns(3)
                            # Función auxiliar para limitar valores a 0-255
                            def clip_uint8(img):
                                return np.clip(img, 0, 255).astype(np.uint8)
                            # 🔴 Canal Rojo
                            with c1:
                                st.subheader("🔴 Canal Rojo")
                                brillo_R = st.slider("Brillo R (+/-)", min_value=-100, max_value=100, value=0)
                                contraste_R = st.slider("Contraste R (x)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
                                previa_Ifilt_r = st.session_state["canalR"].astype(np.float32)
                                previa_Ifilt_r = previa_Ifilt_r * contraste_R + brillo_R
                                previa_Ifilt_r = clip_uint8(previa_Ifilt_r)
                                #figPrevR, axPrevR = plt.subplots(figsize=(7, 5))
                                #axPrevR.imshow(previa_Ifilt_r, cmap='gray')
                                #axPrevR.axis("off")
                                st.write("Img Rojo (Previsualización)")
                                #st.pyplot(figPrevR)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_r),use_container_width=True)
                                #else:
                                #    pass
                                if st.button("Aplicar a Rojo",key="brcon_R"):
                                    img = st.session_state["canalR"].astype(np.float32)
                                    img = img * contraste_R + brillo_R
                                    st.session_state["canalR"] = clip_uint8(img)
                            # 🟢 Canal Verde
                            with c2:
                                st.subheader("🟢 Canal Verde")
                                brillo_G = st.slider("Brillo G (+/-)", min_value=-100, max_value=100, value=0)
                                contraste_G = st.slider("Contraste G (x)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
                                previa_Ifilt_g = st.session_state["canalG"].astype(np.float32)
                                previa_Ifilt_g = previa_Ifilt_g * contraste_G + brillo_G
                                previa_Ifilt_g = clip_uint8(previa_Ifilt_g)
                                #figPrevG, axPrevG = plt.subplots(figsize=(7, 5))
                                #axPrevG.imshow(previa_Ifilt_g, cmap='gray')
                                #axPrevG.axis("off")
                                st.write("Img Verde (Previsualización)")
                                #st.pyplot(figPrevG)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)
                                #else:
                                #    pass
                                if st.button("Aplicar a Verde",key="brcon_G"):
                                    img = st.session_state["canalG"].astype(np.float32)
                                    img = img * contraste_G + brillo_G
                                    st.session_state["canalG"] = clip_uint8(img)
                            # 🔵 Canal Azul
                            with c3:
                                st.subheader("🔵 Canal Azul")
                                brillo_B = st.slider("Brillo B (+/-)", min_value=-100, max_value=100, value=0)
                                contraste_B = st.slider("Contraste B (x)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
                                previa_Ifilt_b = st.session_state["canalB"].astype(np.float32)
                                previa_Ifilt_b = previa_Ifilt_b * contraste_B + brillo_B
                                previa_Ifilt_b = clip_uint8(previa_Ifilt_b)
                                #figPrevB, axPrevB = plt.subplots(figsize=(7, 5))
                                #axPrevB.imshow(previa_Ifilt_b, cmap='gray')
                                #axPrevB.axis("off")
                                st.write("Img Azul (Previsualización)")
                                #st.pyplot(figPrevB)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_b),use_container_width=True)
                                #else:
                                #    pass
                                if st.button("Aplicar a Azul",key="brcon_B"):
                                    img = st.session_state["canalB"].astype(np.float32)
                                    img = img * contraste_B + brillo_B
                                    st.session_state["canalB"] = clip_uint8(img)
                        case "Operaciones morfológicas":
                            st.header("🧱 Operaciones Morfológicas por Canal")
                            # Elegir tamaño del kernel
                            kernel_size = st.slider("Tamaño del kernel estructurante", 1, 15, 3, step=2)
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                            # Crear columnas para cada canal
                            m1, m2, m3 = st.columns(3)
                            # 🔴 Canal Rojo
                            with m1:
                                st.subheader("🔴 Canal Rojo")
                                op_r = st.selectbox("Operación morfológica R", ["Ninguna", "Erosión", "Dilatación", "Apertura", "Cierre"], key="opR")
                                previa_Ifilt_r = st.session_state["canalR"]
                                if op_r == "Erosión":
                                    previa_Ifilt_r = cv2.erode(previa_Ifilt_r, kernel, iterations=1)
                                elif op_r == "Dilatación":
                                    previa_Ifilt_r = cv2.dilate(previa_Ifilt_r, kernel, iterations=1)
                                elif op_r == "Apertura":
                                    previa_Ifilt_r = cv2.morphologyEx(previa_Ifilt_r, cv2.MORPH_OPEN, kernel)
                                elif op_r == "Cierre":
                                    previa_Ifilt_r = cv2.morphologyEx(previa_Ifilt_r, cv2.MORPH_CLOSE, kernel)
                                #figPrevR, axPrevR = plt.subplots(figsize=(7, 5))
                                #axPrevR.imshow(previa_Ifilt_r, cmap='gray')
                                #axPrevR.axis("off")
                                st.write("Img Rojo (Previsualización)")
                                #st.pyplot(figPrevR)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_r),use_container_width=True)
                                #else:
                                #    pass
                                if st.button("Aplicar a Rojo",key="OpR"):
                                    img = st.session_state["canalR"]
                                    if op_r == "Erosión":
                                        img = cv2.erode(img, kernel, iterations=1)
                                    elif op_r == "Dilatación":
                                        img = cv2.dilate(img, kernel, iterations=1)
                                    elif op_r == "Apertura":
                                        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                                    elif op_r == "Cierre":
                                        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                                    st.session_state["canalR"] = img
                            # 🟢 Canal Verde
                            with m2:
                                st.subheader("🟢 Canal Verde")
                                op_g = st.selectbox("Operación morfológica G", ["Ninguna", "Erosión", "Dilatación", "Apertura", "Cierre"], key="opG")
                                previa_Ifilt_g = st.session_state["canalG"]
                                if op_g == "Erosión":
                                    previa_Ifilt_g = cv2.erode(previa_Ifilt_g, kernel, iterations=1)
                                elif op_g == "Dilatación":
                                    previa_Ifilt_g = cv2.dilate(previa_Ifilt_g, kernel, iterations=1)
                                elif op_g == "Apertura":
                                    previa_Ifilt_g = cv2.morphologyEx(previa_Ifilt_g, cv2.MORPH_OPEN, kernel)
                                elif op_g == "Cierre":
                                    previa_Ifilt_g = cv2.morphologyEx(previa_Ifilt_g, cv2.MORPH_CLOSE, kernel)
                                #figPrevG, axPrevG = plt.subplots(figsize=(7, 5))
                                #axPrevG.imshow(previa_Ifilt_g, cmap='gray')
                                #axPrevG.axis("off")
                                st.write("Img Verde (Previsualización)")
                                #st.pyplot(figPrevG)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)  
                                #else:
                                #    pass
                                if st.button("Aplicar a Verde",key="OpG"):
                                    img = st.session_state["canalG"]
                                    if op_g == "Erosión":
                                        img = cv2.erode(img, kernel, iterations=1)
                                    elif op_g == "Dilatación":
                                        img = cv2.dilate(img, kernel, iterations=1)
                                    elif op_g == "Apertura":
                                        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                                    elif op_g == "Cierre":
                                        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                                    st.session_state["canalG"] = img
                            # 🔵 Canal Azul
                            with m3:
                                st.subheader("🔵 Canal Azul")
                                op_b = st.selectbox("Operación morfológica B", ["Ninguna", "Erosión", "Dilatación", "Apertura", "Cierre"], key="opB")
                                previa_Ifilt_b = st.session_state["canalB"]
                                if op_b == "Erosión":
                                    previa_Ifilt_b = cv2.erode(previa_Ifilt_b, kernel, iterations=1)
                                elif op_b == "Dilatación":
                                    previa_Ifilt_b = cv2.dilate(previa_Ifilt_b, kernel, iterations=1)
                                elif op_b == "Apertura":
                                    previa_Ifilt_b = cv2.morphologyEx(previa_Ifilt_b, cv2.MORPH_OPEN, kernel)
                                elif op_b == "Cierre":
                                    previa_Ifilt_b = cv2.morphologyEx(previa_Ifilt_b, cv2.MORPH_CLOSE, kernel)   
                                #figPrevB, axPrevB = plt.subplots(figsize=(7, 5))
                                #axPrevB.imshow(previa_Ifilt_b, cmap='gray')
                                #axPrevB.axis("off")
                                st.write("Img Azul (Previsualización)")
                                #st.pyplot(figPrevB)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_b),use_container_width=True)
                                #else:
                                #    pass
                                if st.button("Aplicar a Azul",key="OpB"):
                                    img = st.session_state["canalB"]
                                    if op_b == "Erosión":
                                        img = cv2.erode(img, kernel, iterations=1)
                                    elif op_b == "Dilatación":
                                        img = cv2.dilate(img, kernel, iterations=1)
                                    elif op_b == "Apertura":
                                        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                                    elif op_b == "Cierre":
                                        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                                    st.session_state["canalB"] = img
                        case "Filtro espacial":
                            st.header("🧹 Filtros Espaciales por Canal")
                            # Filtros predefinidos
                            filtros = {
                                "Suavizado": np.array([[1, 1, 1],
                                                    [1, 1, 1],
                                                    [1, 1, 1]]) / 9,
                                "Realce de bordes": np.array([[0, -1, 0],
                                                            [-1, 5, -1],
                                                            [0, -1, 0]])
                            }
                            # Canales
                            f1, f2, f3 = st.columns(3)
                            # ROJO
                            with f1:
                                st.subheader("🔴 Canal Rojo")
                                tipo_filtro_r = st.selectbox("Filtro R", list(filtros.keys()) + ["Personalizado", "Ninguno"], key="tipoFiltroR")
                                kernel_r = None
                                if tipo_filtro_r in filtros:
                                    kernel_r = filtros[tipo_filtro_r]
                                elif tipo_filtro_r == "Personalizado":
                                    matriz_r = st.text_area(
                                        "Ingrese matriz G (columnas separadas por espacios, filas por enter)", 
                                        "1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9", 
                                        key="matrizR"
                                    )
                                    if matriz_r.strip():  # Evita strings vacíos o espacios
                                        try:
                                            # Intentamos convertir a matriz numérica
                                            kernel_r = np.array([
                                                [float(eval(num)) for num in fila.strip().split()] 
                                                for fila in matriz_r.strip().split("\n")
                                            ])

                                            # Normalizamos la matriz para que actúe como filtro convolucional
                                            sumar = np.sum(kernel_r)
                                            if sumar != 0:
                                                kernel_r = kernel_r / sumar
                                            else:
                                                st.warning("La matriz personalizada tiene suma cero. No se normalizará.")

                                        except Exception as e:
                                            kernel_r = None
                                            st.error(f"Error en la matriz personalizada: {e}")
                                    else:
                                        kernel_r = None
                                        st.warning("No se ingresó ninguna matriz personalizada.")
                                if kernel_r is not None:
                                    previa_Ifilt_r = fn.aplicar_filtro(st.session_state["canalR"], kernel_r)
                                    #figPrevR, axPrevR = plt.subplots(figsize=(7, 5))
                                    #axPrevR.imshow(previa_Ifilt_r, cmap='gray')
                                    #axPrevR.axis("off")
                                    st.write("Img Filtrada Rojo (Previsualización)")
                                    #st.pyplot(figPrevR)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_r),use_container_width=True)
                                    #else:
                                    #    pass

                                if st.button("Aplicar a Rojo", key="aplicarR") and kernel_r is not None:
                                    st.session_state["canalR"] = fn.aplicar_filtro(st.session_state["canalR"], kernel_r)
                            # VERDE
                            with f2:
                                st.subheader("🟢 Canal Verde")
                                tipo_filtro_g = st.selectbox("Filtro G", list(filtros.keys()) + ["Personalizado", "Ninguno"], key="tipoFiltroG")
                                kernel_g = None
                                if tipo_filtro_g in filtros:
                                    kernel_g = filtros[tipo_filtro_g]
                                elif tipo_filtro_g == "Personalizado":
                                    matriz_g = st.text_area(
                                        "Ingrese matriz G (columnas separadas por espacios, filas por enter)", 
                                        "1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9", 
                                        key="matrizG"
                                    )
                                    if matriz_g.strip():  # Evita strings vacíos o espacios
                                        try:
                                            # Intentamos convertir a matriz numérica
                                            kernel_g = np.array([
                                                [float(eval(num)) for num in fila.strip().split()] 
                                                for fila in matriz_g.strip().split("\n")
                                            ])

                                            # Normalizamos la matriz para que actúe como filtro convolucional
                                            sumag = np.sum(kernel_g)
                                            if sumag != 0:
                                                kernel_g = kernel_g / sumag
                                            else:
                                                st.warning("La matriz personalizada tiene suma cero. No se normalizará.")

                                        except Exception as e:
                                            kernel_g = None
                                            st.error(f"Error en la matriz personalizada: {e}")
                                    else:
                                        kernel_g = None
                                        st.warning("No se ingresó ninguna matriz personalizada.")
                                if kernel_g is not None:
                                    previa_Ifilt_g = fn.aplicar_filtro(st.session_state["canalG"], kernel_g)
                                    #figPrevG, axPrevG = plt.subplots(figsize=(7, 5))
                                    #axPrevG.imshow(previa_Ifilt_g, cmap='gray')
                                    #axPrevG.axis("off")
                                    st.write("Img Filtrada Verde (Previsualización)")
                                    #st.pyplot(figPrevG)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)
                                    #else:
                                    #    pass    
                                if st.button("Aplicar a Verde", key="aplicarG") and kernel_g is not None:
                                    st.session_state["canalG"] = fn.aplicar_filtro(st.session_state["canalG"], kernel_g)
                            # AZUL
                            with f3:
                                st.subheader("🔵 Canal Azul")
                                tipo_filtro_b = st.selectbox("Filtro B", list(filtros.keys()) + ["Personalizado", "Ninguno"], key="tipoFiltroB")
                                kernel_b = None
                                if tipo_filtro_b in filtros:
                                    kernel_b = filtros[tipo_filtro_b]
                                elif tipo_filtro_b == "Personalizado":
                                    matriz_b = st.text_area(
                                        "Ingrese matriz R (columnas separadas por espacios, filas por enter)", 
                                        "1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9", 
                                        key="matrizB"
                                    )
                                    if matriz_b.strip():  # Evita strings vacíos o espacios
                                        try:
                                            # Intentamos convertir a matriz numérica
                                            kernel_b = np.array([
                                                [float(eval(num)) for num in fila.strip().split()] 
                                                for fila in matriz_b.strip().split("\n")
                                            ])

                                            # Normalizamos la matriz para que actúe como filtro convolucional
                                            sumab = np.sum(kernel_b)
                                            if sumab != 0:
                                                kernel_b = kernel_b / sumab
                                            else:
                                                st.warning("La matriz personalizada tiene suma cero. No se normalizará.")

                                        except Exception as e:
                                            kernel_b = None
                                            st.error(f"Error en la matriz personalizada: {e}")
                                    else:
                                        kernel_b = None
                                        st.warning("No se ingresó ninguna matriz personalizada.")
                                if kernel_b is not None:
                                    previa_Ifilt_b = fn.aplicar_filtro(st.session_state["canalB"], kernel_b)
                                    #figPrevB, axPrevB = plt.subplots(figsize=(7, 5))
                                    #axPrevB.imshow(previa_Ifilt_b, cmap='gray')
                                    #axPrevB.axis("off")
                                    st.write("Img Filtrada Azul (Previsualización)")
                                    #st.pyplot(figPrevB)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)
                                    #else:
                                    #    pass
                                if st.button("Aplicar a Azul", key="aplicarB") and kernel_b is not None:
                                    st.session_state["canalB"] = fn.aplicar_filtro(st.session_state["canalB"], kernel_b)
                        case "Filtro frecuencial":
                            st.header("🌌 Filtros en el Dominio Frecuencial por Canal")
                            # Canales
                            f1, f2, f3 = st.columns(3)
                            if "F_r" not in st.session_state:
                                st.session_state["F_r"] = None
                            if "F_g" not in st.session_state:
                                st.session_state["F_g"] = None
                            if "F_b" not in st.session_state:
                                st.session_state["F_b"] = None
                            if "Ffilt_r" not in st.session_state:
                                st.session_state["Ffilt_r"] = None
                            if "Ffilt_g" not in st.session_state:
                                st.session_state["Ffilt_g"] = None
                            if "Ffilt_b" not in st.session_state:
                                st.session_state["Ffilt_b"] = None
                            # ROJO
                            with f1:
                                st.subheader("🔴 Canal Rojo")
                                tipo_filtro_r = st.selectbox("Filtro R", ["Pasa bajo", "Pasa alto", "Pasa banda",
                                                                           "Gaussiano", "Pasa alto Gaussiano",
                                                                           "Rampa", "Parzen", "Shepp-Logan",
                                                                           "Hann", "Hamming", "Butterworth"], key="freqTipoR")
                                fc1r = st.slider("Frecuencia relativa de corte 1 (R)", 0.00, 1.00, 0.01, key="fc1r")
                                fc2r = None
                                ordenr = None
                                if tipo_filtro_r == "Pasa banda":
                                    fc2r = st.slider("Frecuencia relativa de corte 2 (R)", fc1r + 0.01, 1.00, fc1r + 0.01, key="fc2r")
                                if tipo_filtro_r == "Butterworth":
                                    ordenr = st.slider("Orden", 1,10,1)
                                # --------- GENERAR VISTA PREVIA AUTOMÁTICA ---------
                                _, _, previa_Ffilt_r, previa_Ifilt_r = fn.filtro_frecuencia(
                                    st.session_state["canalR"], tipo_filtro_r, fc1r, fc2r,ordenr
                                )
                                #figPrevRF, axPrevRF = plt.subplots(figsize=(7,5))
                                #axPrevRF.imshow(previa_Ffilt_r, cmap="gray")
                                #axPrevRF.axis("off")
                                st.write("TF Filtrada Rojo (Previsualización)")
                                #st.pyplot(figPrevRF)
                                #if EDI:    
                                st.image(fn.normalizar_0_255(previa_Ffilt_r),use_container_width=True)
                                #else:
                                #    pass
                                #figPrevR, axPrevR = plt.subplots(figsize=(7, 5))
                                #axPrevR.imshow(previa_Ifilt_r, cmap='gray')
                                #axPrevR.axis("off")
                                st.write("Img Fitrada Rojo (Previsualización)")
                                #st.pyplot(figPrevR)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_r),use_container_width=True)
                                #else:
                                #    pass
                                # ----------------------------------------------------
                                if st.button("Aplicar Frecuencia R",key="AF_r"):  
                                    F_r, mask_r, Ffilt_r, imgR = fn.filtro_frecuencia(
                                        st.session_state["canalR"], tipo_filtro_r, fc1r, fc2r,ordenr
                                    )
                                    st.session_state["F_r"] = F_r
                                    st.session_state["Ffilt_r"] = Ffilt_r
                                    st.session_state["canalR"] = fn.normalizar_0_255(imgR)
                            # VERDE
                            with f2:
                                st.subheader("🟢 Canal Verde")
                                tipo_filtro_g = st.selectbox("Filtro G", ["Pasa bajo", "Pasa alto", "Pasa banda",
                                                                           "Gaussiano", "Pasa alto Gaussiano",
                                                                           "Rampa", "Parzen", "Shepp-Logan",
                                                                           "Hann", "Hamming", "Butterworth"], key="freqTipoG")
                                fc1g = st.slider("Frecuencia relativa de corte 1 (G)", 0.00, 0.99, 0.01, key="fc1g")
                                fc2g = None
                                if tipo_filtro_g == "Pasa banda":
                                    fc2g = st.slider("Frecuencia relativa de corte 2 (G)", fc1g + 0.01, 1.00, fc1g + 0.01, key="fc2g")
                                # --------- GENERAR VISTA PREVIA AUTOMÁTICA ---------
                                _, _, previa_Ffilt_g, previa_Ifilt_g = fn.filtro_frecuencia(
                                    st.session_state["canalG"], tipo_filtro_g, fc1g, fc2g
                                )
                                #figPrevGF, axPrevGF = plt.subplots(figsize=(7,5))
                                #axPrevGF.imshow(previa_Ffilt_g, cmap="gray")
                                #axPrevGF.axis("off")
                                st.write("TF Filtrada Verde (Previsualización)")
                                #st.pyplot(figPrevGF)
                                #if EDI:    
                                st.image(fn.normalizar_0_255(previa_Ffilt_g),use_container_width=True)
                                #else:
                                #    pass
                                #figPrevG, axPrevG = plt.subplots(figsize=(7, 5))
                                #axPrevG.imshow(previa_Ifilt_g, cmap='gray')
                                #axPrevG.axis("off")
                                st.write("Img Filtrada Verde (Previsualización)")
                                #st.pyplot(figPrevG)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)
                                #else:
                                #    pass
                                # ----------------------------------------------------
                                if st.button("Aplicar Frecuencia G", key="AF_g"):
                                    F_g, mask_g, Ffilt_g, imgG = fn.filtro_frecuencia(
                                        st.session_state["canalG"], tipo_filtro_g, fc1g, fc2g
                                    )
                                    st.session_state["F_g"] = F_g
                                    st.session_state["Ffilt_g"] = Ffilt_g
                                    st.session_state["canalG"] =  fn.normalizar_0_255(imgG)
                            # AZUL
                            with f3:
                                st.subheader("🔵 Canal Azul")
                                tipo_filtro_b = st.selectbox("Filtro B", ["Pasa bajo", "Pasa alto", "Pasa banda",
                                                                           "Gaussiano", "Pasa alto Gaussiano",
                                                                           "Rampa", "Parzen", "Shepp-Logan",
                                                                           "Hann", "Hamming", "Butterworth"], key="freqTipoB")
                                fc1b = st.slider("Frecuencia relativa de corte 1 (B)", 0.00, 1.00, 0.01, key="fc1b")
                                fc2b = None
                                if tipo_filtro_b == "Pasa banda":
                                    fc2b = st.slider("Frecuencia relativa de corte 2 (B)", fc1b + 0.01, 1.00, fc1b + 0.01, key="fc2b")
                                # --------- GENERAR VISTA PREVIA AUTOMÁTICA ---------
                                _, _, previa_Ffilt_b, previa_Ifilt_b = fn.filtro_frecuencia(
                                    st.session_state["canalB"], tipo_filtro_b, fc1b, fc2b
                                )
                                #figPrevBF, axPrevBF = plt.subplots(figsize=(7,5))
                                #axPrevBF.imshow(previa_Ffilt_b, cmap="gray")
                                #axPrevBF.axis("off")
                                st.write("TF Filtrada Azul (Previsualización)")
                                #st.pyplot(figPrevBF)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ffilt_b),use_container_width=True)
                                #else:
                                #    pass
                                #figPrevB, axPrevB = plt.subplots(figsize=(7, 5))
                                #axPrevB.imshow(previa_Ifilt_b, cmap='gray')
                                #axPrevB.axis("off")
                                st.write("Img Filtrada Azul (Previsualización)")
                                #st.pyplot(figPrevB)
                                #if EDI:
                                st.image(fn.normalizar_0_255(previa_Ifilt_b),use_container_width=True)
                                #else:
                                #    pass
                                # ----------------------------------------------------
                                if st.button("Aplicar Frecuencia B", key="AF_b"):
                                    F_b, mask_b, Ffilt_b, imgB = fn.filtro_frecuencia(
                                        st.session_state["canalB"], tipo_filtro_b, fc1b, fc2b
                                    )
                                    st.session_state["F_b"] = F_b
                                    st.session_state["Ffilt_b"] = Ffilt_b
                                    st.session_state["canalB"] =  fn.normalizar_0_255(imgB)
                        
                        case "Ecualización":
                                st.header("📊 Ecualización de Histograma")
                                g1, g2, g3 = st.columns(3)
                                with g1:
                                    st.subheader("🔴 Canal Rojo")
                                    tipo_eq_r = st.selectbox("Tipo de ecualización Rojo", ["Global", "Adaptativa (CLAHE)"])
                                    previa_Ifilt_r = st.session_state["canalR"]
                                    if tipo_eq_r == "Global":
                                        previa_Ifilt_r = cv2.equalizeHist(previa_Ifilt_r)
                                    elif tipo_eq_r == "Adaptativa (CLAHE)":
                                        clip_r = st.slider("Clip Limit", 1.0, 10.0, 2.0,key="cl_r")
                                        grid_r = st.slider("Tile Grid Size", 1, 16, 8,key="g_r")
                                        clahe_r = cv2.createCLAHE(clipLimit=clip_r, tileGridSize=(grid_r, grid_r))
                                        previa_Ifilt_r = clahe_r.apply(previa_Ifilt_r)
                                    st.write("Canal Rojo (Ecualizado - Previsualización)")
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_r),use_container_width=True)                                    
                                    if st.button("Aplicar ecualización Rojo"):
                                        st.session_state["canalR"] = previa_Ifilt_r
                                with g2:
                                    st.subheader("🟢 Canal Verde")
                                    tipo_eq_g = st.selectbox("Tipo de ecualización Verde", ["Global", "Adaptativa (CLAHE)"])
                                    previa_Ifilt_g = st.session_state["canalG"]
                                    if tipo_eq_g == "Global":
                                        previa_Ifilt_g = cv2.equalizeHist(previa_Ifilt_g)
                                    elif tipo_eq_g == "Adaptativa (CLAHE)":
                                        clip_g = st.slider("Clip Limit", 1.0, 10.0, 2.0,key="cl_g")
                                        grid_g = st.slider("Tile Grid Size", 1, 16, 8,key="g_g")
                                        clahe_g = cv2.createCLAHE(clipLimit=clip_g, tileGridSize=(grid_g, grid_g))
                                        previa_Ifilt_g = clahe_g.apply(previa_Ifilt_g)
                                    st.write("Canal Verde (Ecualizado - Previsualización)")
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_g),use_container_width=True)                                    
                                    if st.button("Aplicar ecualización Verde"):
                                        st.session_state["canalG"] = previa_Ifilt_g
                                with g3:
                                    st.subheader("🔵 Canal Azul")
                                    tipo_eq_b = st.selectbox("Tipo de ecualización Azul", ["Global", "Adaptativa (CLAHE)"])
                                    previa_Ifilt_b = st.session_state["canalB"]
                                    if tipo_eq_b == "Global":
                                        previa_Ifilt_b = cv2.equalizeHist(previa_Ifilt_b)
                                    elif tipo_eq_b == "Adaptativa (CLAHE)":
                                        clip_b = st.slider("Clip Limit", 1.0, 10.0, 2.0,key="cl_b")
                                        grid_b = st.slider("Tile Grid Size", 1, 16, 8,key="g_b")
                                        clahe_b = cv2.createCLAHE(clipLimit=clip_b, tileGridSize=(grid_b, grid_b))
                                        previa_Ifilt_b = clahe_b.apply(previa_Ifilt_b)
                                    st.write("Canal Azul (Ecualizado - Previsualización)")
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_b),use_container_width=True)                                    
                                    if st.button("Aplicar ecualización Azul"):
                                        st.session_state["canalB"] = previa_Ifilt_b
                    # Mostrar canales modificados
                    #if EDI:
                    qolumna1, qolumna2, qolumna3 = st.columns(3)
                    with qolumna1:
                            #figr, axsr = plt.subplots(1,1,figsize=(7, 5))
                            #axsr.imshow(st.session_state["canalR"], cmap='gray')
                            #axsr.axis('off')
                        st.write("Canal Rojo")
                            #st.pyplot(figr)
                        st.image(fn.normalizar_0_255(st.session_state["canalR"]),use_container_width=True)
                    with qolumna2:
                            #figg, axsg = plt.subplots(figsize=(7, 5))
                            #axsg.imshow(st.session_state["canalG"], cmap='gray')
                            #axsg.axis('off')
                        st.write("Canal Verde")
                            #st.pyplot(figg)
                        st.image(fn.normalizar_0_255(st.session_state["canalG"]),use_container_width=True)
                    with qolumna3:
                            #figb, axsb = plt.subplots(figsize=(7, 5))
                            #axsb.imshow(st.session_state["canalB"], cmap='gray')
                            #axsb.axis('off')
                        st.write("Canal Azul")
                            #st.pyplot(figb)
                        st.image(fn.normalizar_0_255(st.session_state["canalB"]),use_container_width=True)
                    #else:
                    #    pass
                        # Recombinación de canales
                    if all(k in st.session_state for k in ("canalR", "canalG", "canalB")):
                        # Asegurar que todos tengan el mismo tamaño
                        min_shape = np.min([
                            st.session_state["canalR"].shape,
                            st.session_state["canalG"].shape,
                            st.session_state["canalB"].shape
                        ], axis=0)
                        canalR_crop = st.session_state["canalR"][:min_shape[0], :min_shape[1]]
                        canalG_crop = st.session_state["canalG"][:min_shape[0], :min_shape[1]]
                        canalB_crop = st.session_state["canalB"][:min_shape[0], :min_shape[1]]
                        imagen_recombinada = np.stack((canalR_crop, canalG_crop, canalB_crop), axis=-1)
                        if st.button("Recombinar"):
                            columnaf1, columnaf2 = st.columns(2)
                            #if EDI:
                            with columnaf1:
                                fignrecom, axsfnr = plt.subplots(figsize=(15,5))
                                axsfnr.imshow(img_a, cmap='gray')
                                axsfnr.axis('off')
                                st.write("Imagen Original")
                                st.pyplot(fignrecom)                        
                            with columnaf2:
                                figrecom, axsfr = plt.subplots(figsize=(15,5))
                                axsfr.imshow(imagen_recombinada, cmap='gray')
                                axsfr.axis('off')
                                st.write("Imagen Final")
                                st.pyplot(figrecom)
                            st.image(imagen_recombinada, caption="Imagen recombinada", clamp=True,use_container_width=True)
                            #else:
                            #    pass
                        # Opcional: Guardarla en el session_state si después la querés seguir usando
                        st.session_state["img_analisis"] = imagen_recombinada
                else:
                    if img_a.ndim == 2:  # Imagen en escala de grises
                        if img_a.dtype != np.uint8:
                            img_a = cv2.normalize(img_a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                        # Si tiene 3 canales (color), convertir a gris
                        if len(img_a.shape) == 3 and img_a.shape[2] == 3:
                            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
                        if "img_original_gray" not in st.session_state or not np.array_equal(img_a, st.session_state["img_original_gray"]):
                            st.session_state["img_original_gray"] = img_a
                            st.session_state["canalGray"] = img_a
                            st.session_state["F_gray"] = None
                            st.session_state["Ffilt_gray"] = None
                            st.session_state["AplicarFGray"] = None

                        OPERACION = st.selectbox("Elegí una operación:", ["Rotar", "Contraste y brillo", "Operaciones morfológicas",
                                                                    "Filtro espacial", "Filtro frecuencial",
                                                                    "Ecualización"])
                        match OPERACION:
                            case "Rotar":
                                st.header("🎛️ Rotación")
                                c1, c2, c3 = st.columns(3)
                                with c2:
                                    st.subheader("Canal Gris")
                                    angulo_gray = st.slider("Ángulo de rotación R", min_value=-180, max_value=180, value=0, step=1)
                                    previa_Ifilt_gray = scipy.ndimage.rotate(st.session_state["canalGray"], angle=angulo_gray, reshape=False, mode='nearest')
                                    #figPrevgray, axPrevgray = plt.subplots(figsize=(7, 5))
                                    #axPrevgray.imshow(previa_Ifilt_gray, cmap='gray')
                                    #axPrevgray.axis("off")
                                    st.write("Img Gris (Previsualización)")
                                    #st.pyplot(figPrevgray)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_gray),use_container_width=True)
                                    #else:
                                    #    pass
                                    if st.button("Rotar"):
                                        st.session_state["canalGray"] = scipy.ndimage.rotate(st.session_state["canalGray"], angle=angulo_gray, reshape=False, mode='nearest')
                            case "Contraste y brillo":
                                st.header("✨ Ajustes de Brillo y Contraste")
                                # Crear columnas para cada canal
                                c1, c2, c3 = st.columns(3)
                                # Función auxiliar para limitar valores a 0-255
                                def clip_uint8(img):
                                    return np.clip(img, 0, 255).astype(np.uint8)
                                # 🔴 Canal Rojo
                                with c2:
                                    st.subheader("Canal Gris")
                                    brillo_gray = st.slider("Brillo Gris (+/-)", min_value=-100, max_value=100, value=0)
                                    contraste_gray = st.slider("Contraste Gris (x)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
                                    previa_Ifilt_gray = st.session_state["canalGray"]#.astype(np.float32)
                                    previa_Ifilt_gray = previa_Ifilt_gray * contraste_gray + brillo_gray
                                    previa_Ifilt_gray = clip_uint8(previa_Ifilt_gray)
                                    #figPrevgray, axPrevgray = plt.subplots(figsize=(7, 5))
                                    #axPrevgray.imshow(previa_Ifilt_gray, cmap='gray')
                                    #axPrevgray.axis("off")
                                    st.write("Img Gris (Previsualización)")
                                    #st.pyplot(figPrevgray)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_gray),use_container_width=True)
                                    #else:
                                    #    pass
                                    if st.button("Aplicar",key="brcon_gray"):
                                        img = st.session_state["canalGray"].astype(np.float32)
                                        img = img * contraste_gray + brillo_gray
                                        st.session_state["canalGray"] = clip_uint8(img)
                            case "Operaciones morfológicas":
                                st.header("🧱 Operaciones Morfológicas")
                                # Elegir tamaño del kernel
                                kernel_size = st.slider("Tamaño del kernel estructurante", 1, 15, 3, step=2)
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                                # Crear columnas para cada canal
                                m1, m2, m3 = st.columns(3)
                                # 🔴 Canal Rojo
                                with m2:
                                    st.subheader("Canal Gris")
                                    op_gray = st.selectbox("Operación morfológica Gris", ["Ninguna", "Erosión", "Dilatación", "Apertura", "Cierre"], key="opR")
                                    previa_Ifilt_gray = st.session_state["canalGray"]
                                    if op_gray == "Erosión":
                                        previa_Ifilt_gray = cv2.erode(previa_Ifilt_gray, kernel, iterations=1)
                                    elif op_gray == "Dilatación":
                                        previa_Ifilt_gray = cv2.dilate(previa_Ifilt_gray, kernel, iterations=1)
                                    elif op_gray == "Apertura":
                                        previa_Ifilt_gray = cv2.morphologyEx(previa_Ifilt_gray, cv2.MORPH_OPEN, kernel)
                                    elif op_gray == "Cierre":
                                        previa_Ifilt_gray = cv2.morphologyEx(previa_Ifilt_gray, cv2.MORPH_CLOSE, kernel)
                                    #figPrevgray, axPrevgray = plt.subplots(figsize=(7, 5))
                                    #axPrevgray.imshow(previa_Ifilt_gray, cmap='gray')
                                    #axPrevgray.axis("off")
                                    st.write("Img (Previsualización)")
                                    #st.pyplot(figPrevgray)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_gray),use_container_width=True)
                                    #else:
                                    #    pass
                                    if st.button("Aplicar",key="OpGray"):
                                        img = st.session_state["canalGray"]
                                        if op_gray == "Erosión":
                                            img = cv2.erode(img, kernel, iterations=1)
                                        elif op_gray == "Dilatación":
                                            img = cv2.dilate(img, kernel, iterations=1)
                                        elif op_gray == "Apertura":
                                            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                                        elif op_gray == "Cierre":
                                            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                                        st.session_state["canalGray"] = img
                            case "Filtro espacial":
                                st.header("🧹 Filtros Espaciales por Canal")
                                # Filtros predefinidos
                                filtros = {
                                    "Suavizado": np.array([[1, 1, 1],
                                                        [1, 1, 1],
                                                        [1, 1, 1]]) / 9,
                                    "Realce de bordes": np.array([[0, -1, 0],
                                                                [-1, 5, -1],
                                                                [0, -1, 0]])
                                }
                                # Canales
                                f1, f2, f3 = st.columns(3)
                                # ROJO
                                with f2:
                                    st.subheader("Canal Gris")
                                    tipo_filtro_gray = st.selectbox("Filtro Gris", list(filtros.keys()) + ["Personalizado", "Ninguno"], key="tipoFiltroR")
                                    kernel_gray = None
                                    if tipo_filtro_gray in filtros:
                                        kernel_gray = filtros[tipo_filtro_gray]
                                    elif tipo_filtro_gray == "Personalizado":
                                        matriz_gray = st.text_area(
                                            "Ingrese matriz (columnas separadas por espacios, filas por enter)", 
                                            "1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9", 
                                            key="matrizGray"
                                        )
                                        if matriz_gray.strip():  # Evita strings vacíos o espacios
                                            try:
                                                # Intentamos convertir a matriz numérica
                                                kernel_gray = np.array([
                                                    [float(eval(num)) for num in fila.strip().split()] 
                                                    for fila in matriz_gray.strip().split("\n")
                                                ])

                                                # Normalizamos la matriz para que actúe como filtro convolucional
                                                suma = np.sum(kernel_gray)
                                                if suma != 0:
                                                    kernel_gray = kernel_gray / suma
                                                else:
                                                    st.warning("La matriz personalizada tiene suma cero. No se normalizará.")

                                            except Exception as e:
                                                kernel_gray = None
                                                st.error(f"Error en la matriz personalizada: {e}")
                                        else:
                                            kernel_gray = None
                                            st.warning("No se ingresó ninguna matriz personalizada.")
                                    if kernel_gray is not None:
                                        previa_Ifilt_gray = fn.aplicar_filtro(st.session_state["canalGray"], kernel_gray)
                                        #figPrevgray, axPrevgray = plt.subplots(figsize=(7, 5))
                                        #axPrevgray.imshow(previa_Ifilt_gray, cmap='gray')
                                        #axPrevgray.axis("off")
                                        st.write("Img Filtrada (Previsualización)")
                                        #st.pyplot(figPrevgray)
                                        #if EDI:
                                        st.image(fn.normalizar_0_255(previa_Ifilt_gray),use_container_width=True)
                                        #else:
                                        #    pass
                                    else:
                                        st.info("No se aplicó ningún filtro personalizado.")

                                    if st.button("Aplicar", key="aplicarGray") and kernel_gray is not None:
                                        st.session_state["canalGray"] = fn.aplicar_filtro(st.session_state["canalGray"], kernel_gray)
                            
                            case "Filtro frecuencial":
                                st.header("🌌 Filtros en el Dominio Frecuencial")
                                # Canales
                                f1, f2,f3 = st.columns(3)
                                # ROJO
                                with f2:
                                    st.subheader("Canal Gris")
                                    tipo_filtro_gray = st.selectbox("Filtro R", ["Pasa bajo", "Pasa alto", "Pasa banda",
                                                                            "Gaussiano", "Pasa alto Gaussiano",
                                                                            "Rampa", "Parzen", "Shepp-Logan",
                                                                            "Hann", "Hamming", "Butterworth"], key="freqTipogray")
                                    fc1gray = st.slider("Frecuencia relativa de corte 1 (R)", 0.00, 1.00, 0.01, key="fc1gray")
                                    fc2gray = None
                                    ordengray = None
                                    if tipo_filtro_gray == "Pasa banda":
                                        fc2gray = st.slider("Frecuencia relativa de corte 2 (R)", fc1gray + 0.01, 1.00, fc1gray + 0.01, key="fc2gray")
                                    if tipo_filtro_gray == "Butterworth":
                                        ordengray = st.slider("Orden", 1,10,1)
                                    # --------- GENERAR VISTA PREVIA AUTOMÁTICA ---------
                                    _, _, previa_Ffilt_gray, previa_Ifilt_gray = fn.filtro_frecuencia(
                                        st.session_state["canalGray"], tipo_filtro_gray, fc1gray, fc2gray,ordengray
                                    )
                                    #figPrevgrayF, axPrevgrayF = plt.subplots(figsize=(7,5))
                                    #axPrevgrayF.imshow(previa_Ffilt_gray, cmap="gray")
                                    #axPrevgrayF.axis("off")
                                    st.write("TF Filtrada (Previsualización)")
                                    #st.pyplot(figPrevgrayF)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ffilt_gray),use_container_width=True)
                                    #else:
                                    #    pass
                                    #figPrevgray, axPrevgray = plt.subplots(figsize=(7, 5))
                                    #axPrevgray.imshow(previa_Ifilt_gray, cmap='gray')
                                    #axPrevgray.axis("off")
                                    st.write("Img Fitrada (Previsualización)")
                                    #st.pyplot(figPrevgray)
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_gray))
                                    #else:
                                    #    pass
                                    # ----------------------------------------------------
                                    if st.button("AplicarFGray",key="AF_gray"):  
                                        F_gray, mask_gray, Ffilt_gray, imggray = fn.filtro_frecuencia(
                                            st.session_state["canalGray"], tipo_filtro_gray, fc1gray, fc2gray,ordengray
                                        )
                                        st.session_state["F_gray"] = F_gray
                                        st.session_state["Ffilt_gray"] = Ffilt_gray
                                        st.session_state["canalGray"] = fn.normalizar_0_255(imggray)
                            
                            case "Ecualización":
                                st.header("📊 Ecualización de Histograma")
                                tipo_eq_gray = st.selectbox("Tipo de ecualización", ["Global", "Adaptativa (CLAHE)"])
                                previa_Ifilt_gray = st.session_state["canalGray"]
                                if tipo_eq_gray == "Global":
                                    previa_Ifilt_gray = cv2.equalizeHist(previa_Ifilt_gray)
                                elif tipo_eq_gray == "Adaptativa (CLAHE)":
                                    clip_gray = st.slider("Clip Limit", 1.0, 10.0, 2.0,key="cl_gray")
                                    grid_gray = st.slider("Tile Grid Size", 1, 16, 8,key="g_gray")
                                    clahe_gray = cv2.createCLAHE(clipLimit=clip_gray, tileGridSize=(grid_gray, grid_gray))
                                    previa_Ifilt_gray = clahe_gray.apply(previa_Ifilt_gray)
                                g1, g2, g3 = st.columns(3)
                                with g2:
                                    st.subheader("Canal Gris")
                                    st.write("Img Gris (Ecualizada - Previsualización)")
                                    #if EDI:
                                    st.image(fn.normalizar_0_255(previa_Ifilt_gray),use_container_width=True)
                                    
                                    if st.button("Aplicar ecualización"):
                                        st.session_state["canalGray"] = previa_Ifilt_gray
                        # Mostrar canales modificados
                        qolumna1, qolumna2, qolumna3 = st.columns(3)
                        with qolumna2:
                            #figgray, axsgray = plt.subplots(1,1,figsize=(7, 5))
                            #axsgray.imshow(st.session_state["canalGray"], cmap='gray')
                            #axsgray.axis('off')
                            st.write("Canal Gris")
                            #st.pyplot(figgray)
                            #if EDI:
                            st.image(fn.normalizar_0_255(st.session_state["canalGray"]),use_container_width=True)
                            #else:
                            #    pass
                        columnaf1, columnaf2 = st.columns(2)
                        with columnaf1:
                            fignrecom, axsfnr = plt.subplots(figsize=(15,5))
                            axsfnr.imshow(img_a, cmap='gray')
                            axsfnr.axis('off')
                            st.write("Imagen Original")
                            #if EDI:
                            st.pyplot(fignrecom)
                            #else:
                            #    pass                        
                        with columnaf2:
                            imagen_gris = st.session_state["canalGray"]
                            figrecomgray, axsfgray = plt.subplots(figsize=(15,5))
                            axsfgray.imshow(imagen_gris, cmap='gray')
                            axsfgray.axis('off')
                            st.write("Imagen Final")
                            #if EDI:
                            st.pyplot(figrecomgray)
                            #else:
                            #    pass
                        st.image(fn.normalizar_0_255(imagen_gris), caption="Imagen Gris", clamp=True,use_container_width=True)
                    # Opcional: Guardarla en el session_state si después la querés seguir usando
                        
                        st.session_state["img_analisis"] = imagen_gris
        
        #### -----------------
        #### Calibración
        #### -----------------

        with tabsp[3]:
            st.subheader("📏 Calibración de escala")
            if "escala_mm_por_pixel" not in st.session_state:
                st.session_state["escala_mm_por_pixel"] = None

            #if st.session_state["escala_mm_por_pixel"] is not None:
            #    st.info(f"📏 Escala actual: {st.session_state['escala_mm_por_pixel']:.4f} mm/píxel")

            metodo = st.radio(
                "Seleccioná el método de calibración:",
                ("Ingresar valor manualmente", "Dibujar una línea"),
                horizontal=True
            )

            if metodo == "Ingresar valor manualmente":
                mm_por_pixel_manual = st.number_input(
                    "🔢 Ingresá el valor de milímetros por píxel:",
                    min_value=0.0001,
                    format="%.4f"
                )

                if mm_por_pixel_manual > 0:
                    st.session_state["escala_mm_por_pixel"] = mm_por_pixel_manual
                    st.success(f"🧮 Escala establecida: {mm_por_pixel_manual:.4f} mm/píxel")

            else:  # Dibujar línea
                st.markdown("🖍️ Dibujá una línea sobre un objeto de longitud conocida en la imagen.")
                img_c = None
                if img_t is not None:
                    CEGC = st.radio("¿Con cuál imagen deseas calibrar?", ["Imagen Original", "Imagen Ventaneada", "Imagen editada"],horizontal=True,key="CEGC")
                    if CEGC == "Imagen Original":
                        if img_t is not None:
                            img_c = np.array(img_t)
                        else:
                            img_c = None
                    elif CEGC =="Imagen Ventaneada":
                        if img_v is not None:
                            img_c = np.array(img_v)
                        else:
                            img_c = None
                    elif CEGC == "Imagen editada":
                        img_c = st.session_state.get("img_analisis", None)
                if img_c is not None:
                    # Preprocesamiento de imagen
                    img_c_view = img_c.copy()
                    vmin = np.min(img_c_view)
                    vmax = np.max(img_c_view)
                    img_view = np.clip((img_c_view - vmin) / (vmax - vmin), 0, 1)
                    img_view = (img_view * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_view).convert("RGB")

                    # Redimensionado
                    width_orig, height_orig = img_pil.size
                    max_canvas_size = 512
                    scale = min(max_canvas_size / width_orig, max_canvas_size / height_orig)
                    new_width = int(width_orig * scale)
                    new_height = int(height_orig * scale)
                    img_resized = img_pil.resize((new_width, new_height))

                    # Canvas
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=1,
                        stroke_color="cyan",
                        background_image=img_resized,
                        update_streamlit=True,
                        height=new_height,
                        width=new_width,
                        drawing_mode="line",
                        key="calibracion_canvas"
                    )

                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        if len(objects) > 0 and objects[0]["type"] == "line":
                            obj = objects[0]

                            x1 = obj["left"] + obj["x1"]
                            y1 = obj["top"] + obj["y1"]
                            x2 = obj["left"] + obj["x2"]
                            y2 = obj["top"] + obj["y2"]

                            x1_orig, y1_orig = x1 / scale, y1 / scale
                            x2_orig, y2_orig = x2 / scale, y2 / scale

                            distancia_px = hypot(x2_orig - x1_orig, y2_orig - y1_orig)
                            st.write(f"📐 Distancia en píxeles: {distancia_px:.2f}")

                            longitud_real = st.number_input(
                                "📏 Longitud real de la línea (en mm)",
                                min_value=0.001,
                                format="%.3f"
                            )

                            if longitud_real > 0:
                                mm_por_pixel = longitud_real / distancia_px
                                st.session_state["escala_mm_por_pixel"] = mm_por_pixel
                                st.success(f"🧮 Tamaño de píxel: {mm_por_pixel:.4f} mm/píxel")

                                # Mostrar línea en imagen original
                                img_con_linea = img_pil.copy()
                                draw = ImageDraw.Draw(img_con_linea)
                                draw.line([(x1_orig, y1_orig), (x2_orig, y2_orig)], fill="red", width=3)
                                st.image(img_con_linea, caption="📸 Línea proyectada en imagen original")            

   
        
        #### -----------------
        #### Análisis
        #### -----------------

        #with st.expander("Análisis de imágenes"):
        with tabsp[4]:
            st.subheader("Analizador de imágenes")
            # Función para capturar los puntos de la imagen
            def select_points(event, x, y, flags, param):
                global points
                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(points) < 2:
                        points.append((x, y))

            # Almacenamos los puntos seleccionados
            points = []

            # Configurar la ventana de OpenCV para seleccionar los puntos
            #cv2.namedWindow('Imagen con OpenCV')
            #cv2.setMouseCallback('Imagen con OpenCV', select_points)
            img_np = st.session_state.get("img_analisis", None)
            
            if img_np is not None:
                # Convertir BGR a RGB para mostrar correctamente
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            if img_np is not None and img_a is not None:
                # Detectar si se subió una nueva imagen
                imagen_actual_shape = img_a.shape  # como firma simple de comparación

                if "imagen_anterior_shape" not in st.session_state:
                    st.session_state["imagen_anterior_shape"] = imagen_actual_shape

                if st.session_state["imagen_anterior_shape"] != imagen_actual_shape:
                    # Actualizar la imagen registrada
                    st.session_state["imagen_anterior_shape"] = imagen_actual_shape

                    # 🔁 Resetear todas las variables que dependían de la imagen anterior
                     # 🔁 Resetear estado relacionado a la imagen anterior
                    claves_a_resetear = [
                        "x1", "x2", "y1", "y2", "escala_mm_por_pixel",
                        "calibracion_canvas", "perfil_canvas",
                        "canvas_width", "canvas_height", "scale"
                    ]
                    for clave in claves_a_resetear:
                        st.session_state.pop(clave, None)
                tolum1,tolum2 = st.columns(2)
                with tolum1:
                    tipo = st.selectbox("Elegí el tipo de análisis", ["Histograma", "Perfil", "ROIs"])
                match tipo:
                    case "Trazar ejes":
                        
                        st.subheader("➕ Trazar ejes coordenados sobre la imagen")

                        # Escalado para visualización
                        img_np_view = img_np.copy()
                        vmin = np.min(img_np_view)
                        vmax = np.max(img_np_view)
                        img_view = np.clip((img_np_view - vmin) / (vmax - vmin), 0, 1)
                        img_view = (img_view * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_view).convert("RGB")

                        # Redimensionamos para canvas
                        height_orig, width_orig = img_np.shape[:2]
                        max_canvas_size = 512
                        scale = min(max_canvas_size / width_orig, max_canvas_size / height_orig)
                        canvas_width = int(width_orig * scale)
                        canvas_height = int(height_orig * scale)
                        img_resized = img_pil.resize((canvas_width, canvas_height))

                        # Controles para el centro y el ángulo
                        col1, col2 = st.columns(2)
                        with col1:
                            cx = st.slider("Centro X (en canvas)", 0, canvas_width, canvas_width // 2)
                            cy = st.slider("Centro Y (en canvas)", 0, canvas_height, canvas_height // 2)
                        with col2:
                            angulo = st.slider("Ángulo de los ejes (°)", -180, 180, 0)

                        #🧩 2. Calculamos puntos y dibujamos los ejes

                        # Longitud visual de cada eje (en canvas)
                        eje_len = min(canvas_width, canvas_height) // 3

                        # Ángulo en radianes
                        theta = np.deg2rad(angulo)

                        # Vector unitario dirección del eje X
                        dx = np.cos(theta)
                        dy = np.sin(theta)

                        # Extremos del eje X
                        x1_x = cx - dx * eje_len
                        y1_x = cy - dy * eje_len
                        x2_x = cx + dx * eje_len
                        y2_x = cy + dy * eje_len

                        # Eje Y es perpendicular (90°)
                        dx_perp = -dy
                        dy_perp = dx

                        x1_y = cx - dx_perp * eje_len
                        y1_y = cy - dy_perp * eje_len
                        x2_y = cx + dx_perp * eje_len
                        y2_y = cy + dy_perp * eje_len

                        # Dibujamos con OpenCV sobre copia de imagen canvas
                        ejes_img = np.array(img_resized.copy())
                        ejes_img = cv2.line(ejes_img, (int(x1_x), int(y1_x)), (int(x2_x), int(y2_x)), (255, 0, 0), 1)  # Eje X - rojo
                        ejes_img = cv2.line(ejes_img, (int(x1_y), int(y1_y)), (int(x2_y), int(y2_y)), (0, 255, 0), 1)  # Eje Y - verde
                        ejes_img_pil = Image.fromarray(ejes_img)

                        # Mostramos
                        st.image(ejes_img_pil, caption="Ejes de coordenadas trazados sobre la imagen (visualización)")
                    case "Histograma":
                        st.subheader("📊 Histograma")

                        log_escala = st.checkbox("Usar escala logarítmica (eje Y)", value=False)

                        # Aplanamos y limpiamos la imagen
                        datos_hist = img_np.ravel()
                        datos_hist = datos_hist[np.isfinite(datos_hist)]

                        # Detectar rango dinámico
                        min_val_hist = datos_hist.min()
                        max_val_hist = datos_hist.max()
                        st.write(f"🔢 Rango de valores: {min_val_hist:.2f} a {max_val_hist:.2f}")

                        # Escala de grises
                        if img_np.ndim == 2:
                            bin_centers, hist = fn.plot_histograma_estilo_matlab(datos_hist, color='gray')
                            fig, ax = plt.subplots()
                            ax.vlines(bin_centers, 0, hist, color='gray', lw=1)
                            ax.set_title("Histograma - Escala de grises")
                            ax.set_xlabel("Valor de píxel")
                            ax.set_ylabel("Frecuencia")
                            if log_escala:
                                ax.set_yscale('log')
                            st.pyplot(fig)

                        # Imagen RGB con selección de canal
                        elif img_np.ndim == 3 and img_np.shape[2] == 3:
                            opcion = st.selectbox(
                                "Canal a mostrar",
                                options=["Todos", "Rojo", "Verde", "Azul", "Promedio RGB"],
                                index=0
                            )
                            fig, ax = plt.subplots()
                            if opcion == "Todos":
                                colores = ['r', 'g', 'b']
                                etiquetas = ['Rojo', 'Verde', 'Azul']
                                for i, c in enumerate(colores):
                                    canal = img_np[:, :, i].ravel()
                                    bin_centers, hist = fn.plot_histograma_estilo_matlab(canal, color=c)
                                    ax.vlines(bin_centers, 0, hist, color=c, lw=1, label=etiquetas[i])
                                ax.legend()
                            elif opcion in ["Rojo", "Verde", "Azul"]:
                                canal_idx = {"Rojo": 0, "Verde": 1, "Azul": 2}[opcion]
                                color = {"Rojo": 'r', "Verde": 'g', "Azul": 'b'}[opcion]
                                canal = img_np[:, :, canal_idx].ravel()
                                bin_centers, hist = fn.plot_histograma_estilo_matlab(canal, color=color)
                                ax.vlines(bin_centers, 0, hist, color=color, lw=1)
                            elif opcion == "Promedio RGB":
                                promedio = img_np.mean(axis=2).ravel()
                                bin_centers, hist = fn.plot_histograma_estilo_matlab(promedio, color='gray')
                                ax.vlines(bin_centers, 0, hist, color='gray', lw=1)
                            ax.set_title(f"Histograma - {opcion}")
                            ax.set_xlabel("Valor de píxel")
                            ax.set_ylabel("Frecuencia")
                            if log_escala:
                                ax.set_yscale('log')
                            st.pyplot(fig)

                        # Imagen RGBA (sin selector aún, pero puede agregarse igual)
                        elif img_np.ndim == 3 and img_np.shape[2] == 4:
                            fig, ax = plt.subplots()
                            colores = ['r', 'g', 'b', 'k']
                            etiquetas = ['Rojo', 'Verde', 'Azul', 'Alfa']
                            for i, c in enumerate(colores):
                                canal = img_np[:, :, i].ravel()
                                bin_centers, hist = fn.plot_histograma_estilo_matlab(canal, color=c)
                                ax.vlines(bin_centers, 0, hist, color=c, lw=1, label=etiquetas[i])
                            ax.set_title("Histograma por canal (RGBA)")
                            ax.set_xlabel("Valor de píxel")
                            ax.set_ylabel("Frecuencia")
                            if log_escala:
                                ax.set_yscale('log')
                            ax.legend()
                            st.pyplot(fig)

                        else:
                            st.warning("No se pudo calcular el histograma: formato de imagen no compatible.")

                    case "Calibración":
                        st.subheader("📏 Calibración de escala")
                        if "escala_mm_por_pixel" not in st.session_state:
                            st.session_state["escala_mm_por_pixel"] = None

                        #if st.session_state["escala_mm_por_pixel"] is not None:
                        #    st.info(f"📏 Escala actual: {st.session_state['escala_mm_por_pixel']:.4f} mm/píxel")

                        metodo = st.radio(
                            "Seleccioná el método de calibración:",
                            ("Ingresar valor manualmente", "Dibujar una línea"),
                            horizontal=True
                        )

                        if metodo == "Ingresar valor manualmente":
                            mm_por_pixel_manual = st.number_input(
                                "🔢 Ingresá el valor de milímetros por píxel:",
                                min_value=0.0001,
                                format="%.4f"
                            )

                            if mm_por_pixel_manual > 0:
                                st.session_state["escala_mm_por_pixel"] = mm_por_pixel_manual
                                st.success(f"🧮 Escala establecida: {mm_por_pixel_manual:.4f} mm/píxel")

                        else:  # Dibujar línea
                            st.markdown("🖍️ Dibujá una línea sobre un objeto de longitud conocida en la imagen.")

                            # Preprocesamiento de imagen
                            img_np_view = img_np.copy()
                            vmin = np.min(img_np_view)
                            vmax = np.max(img_np_view)
                            img_view = np.clip((img_np_view - vmin) / (vmax - vmin), 0, 1)
                            img_view = (img_view * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_view).convert("RGB")

                            # Redimensionado
                            width_orig, height_orig = img_pil.size
                            max_canvas_size = 512
                            scale = min(max_canvas_size / width_orig, max_canvas_size / height_orig)
                            new_width = int(width_orig * scale)
                            new_height = int(height_orig * scale)
                            img_resized = img_pil.resize((new_width, new_height))

                            # Canvas
                            canvas_result = st_canvas(
                                fill_color="rgba(255, 0, 0, 0.3)",
                                stroke_width=1,
                                stroke_color="cyan",
                                background_image=img_resized,
                                update_streamlit=True,
                                height=new_height,
                                width=new_width,
                                drawing_mode="line",
                                key="calibracion_canvas"
                            )

                            if canvas_result.json_data is not None:
                                objects = canvas_result.json_data["objects"]
                                if len(objects) > 0 and objects[0]["type"] == "line":
                                    obj = objects[0]

                                    x1 = obj["left"] + obj["x1"]
                                    y1 = obj["top"] + obj["y1"]
                                    x2 = obj["left"] + obj["x2"]
                                    y2 = obj["top"] + obj["y2"]

                                    x1_orig, y1_orig = x1 / scale, y1 / scale
                                    x2_orig, y2_orig = x2 / scale, y2 / scale

                                    distancia_px = hypot(x2_orig - x1_orig, y2_orig - y1_orig)
                                    st.write(f"📐 Distancia en píxeles: {distancia_px:.2f}")

                                    longitud_real = st.number_input(
                                        "📏 Longitud real de la línea (en mm)",
                                        min_value=0.001,
                                        format="%.3f"
                                    )

                                    if longitud_real > 0:
                                        mm_por_pixel = longitud_real / distancia_px
                                        st.session_state["escala_mm_por_pixel"] = mm_por_pixel
                                        st.success(f"🧮 Tamaño de píxel: {mm_por_pixel:.4f} mm/píxel")

                                        # Mostrar línea en imagen original
                                        img_con_linea = img_pil.copy()
                                        draw = ImageDraw.Draw(img_con_linea)
                                        draw.line([(x1_orig, y1_orig), (x2_orig, y2_orig)], fill="red", width=3)
                                        st.image(img_con_linea, caption="📸 Línea proyectada en imagen original")
                    case "Perfil":
                        st.subheader("📈 Perfil de Intensidad con Canvas")
                        # Selector de modo
                        modo_dibujo = st.radio("🖌️ Modo de dibujo:", ["Dibujar perfil", "Marcar centro"], horizontal=True)

                        modo = "rect" if modo_dibujo == "Marcar centro" else "line"

                        # Imagen para visualizar
                        img_np_view = img_np.copy()
                        vmin = np.min(img_np_view)
                        vmax = np.max(img_np_view)
                        img_view = np.clip((img_np_view - vmin) / (vmax - vmin), 0, 1)
                        img_view = (img_view * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_view).convert("RGB")

                        # Ajustar tamaño del canvas y escalas independientes
                        height_orig, width_orig = img_np.shape[:2]
                        max_canvas_size = 512
                        scale_x = min(max_canvas_size / width_orig, max_canvas_size / height_orig)
                        scale_y = scale_x  # Para mantener proporción 1:1 (esto lo puedes cambiar si querés soporte para escalado no uniforme)
                        canvas_width = int(width_orig * scale_x)
                        canvas_height = int(height_orig * scale_y)
                        img_resized = img_pil.resize((canvas_width, canvas_height))

                        # Canvas único
                        st.markdown("🖱️ Dibujá un **rectángulo** para el centro y una **línea** para el perfil. Se conservará uno solo de cada tipo.")
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=1,
                            stroke_color="lime",
                            background_image=img_resized,
                            update_streamlit=True,
                            height=canvas_height,
                            width=canvas_width,
                            drawing_mode=modo,
                            key="canvas_modo_unico"
                        )

                        # Inicializar variables
                        centro = None
                        x1_orig = y1_orig = x2_orig = y2_orig = None

                        objetos = canvas_result.json_data["objects"] if canvas_result.json_data and "objects" in canvas_result.json_data else []
                        # Procesar datos del canvas
                        #if canvas_result.json_data and "objects" in canvas_result.json_data:
                        #    objetos = canvas_result.json_data["objects"]
                            # Detectar último círculo
                            # Detectar último rectángulo
                        rectangulos = [obj for obj in objetos if obj["type"] == "rect"]
                        if rectangulos:
                            rect = rectangulos[-1]
                            rect_left = rect["left"]
                            rect_top = rect["top"]
                            rect_width = rect["width"] * rect.get("scaleX", 1.0)
                            rect_height = rect["height"] * rect.get("scaleY", 1.0)

                            centro_x = (rect_left + rect_width / 2) / scale_x
                            centro_y = (rect_top + rect_height / 2) / scale_y
                            centro = np.array([centro_y, centro_x])
                            st.success(f"✅ Centro definido en: ({centro_x:.1f}, {centro_y:.1f})")

                        # Detectar última línea
                        lineas = [obj for obj in objetos if obj["type"] == "line"]
                        if lineas:
                            obj = lineas[-1]
                            x1 = obj["left"] + obj["x1"]
                            y1 = obj["top"] + obj["y1"]
                            x2 = obj["left"] + obj["x2"]
                            y2 = obj["top"] + obj["y2"]

                            x1_orig, y1_orig = x1 / scale_x, y1 / scale_y
                            x2_orig, y2_orig = x2 / scale_x, y2 / scale_y

                            rr, cc = line(int(round(y1_orig)), int(round(x1_orig)), int(round(y2_orig)), int(round(x2_orig)))
                            rr = np.clip(rr, 0, img_np.shape[0] - 1)
                            cc = np.clip(cc, 0, img_np.shape[1] - 1)
                            profile = img_np[rr, cc]

                            coords = np.stack([rr, cc], axis=1)

                            if centro is not None:
                                vec_line = np.array([y2_orig - y1_orig, x2_orig - x1_orig])
                                vec_line = vec_line / np.linalg.norm(vec_line)
                                vec_centro_a_puntos = coords - centro
                                distances = vec_centro_a_puntos @ vec_line
                            else:
                                diffs = np.sqrt(np.diff(rr)**2 + np.diff(cc)**2)
                                distances = np.concatenate(([0], np.cumsum(diffs)))

                            mm_per_pixel = st.session_state.get("escala_mm_por_pixel", 1.0)
                            mostrar_mm = st.checkbox("Mostrar posición en milímetros", value=False)
                            usar_curva_filtrada_perfil = st.checkbox("Usar curva filtrada",key="UCFP")
                            if mostrar_mm:
                                distances *= mm_per_pixel
                                unidad = "mm"
                            else:
                                unidad = "píxeles"

                            st.markdown("### Medir sobre el perfil")
                            col1, col2 = st.columns(2)

                            with col1:
                                eje_min, eje_max = float(distances[0]), float(distances[-1])
                                if eje_min < eje_max:
                                    x_ini, x_fin = st.slider(
                                        f"Seleccioná dos posiciones en {unidad}",
                                        min_value=eje_min,
                                        max_value=eje_max,
                                        value=(eje_min, eje_max),
                                        step=(0.01),#mm_per_pixel / 100 if mostrar_mm else
                                        format="%.2f"
                                    )
                                    delta = abs(x_fin - x_ini)
                                else:
                                    st.error("No se pudo trazar un perfil")
                                    st.stop()
                                #st.info(f"📏 Distancia seleccionada: **{delta:.2f} {unidad}**")
                                if profile.ndim == 2 and profile.shape[1] == 3:
                                    # 📌 Selector para elegir qué canal mostrar
                                    opcion_canal = st.selectbox("Mostrar canal", ["Todos", "Rojo", "Verde", "Azul", "Promedio"])

                            with col2:
                                # Armo el DataFrame con las columnas según el perfil
                                if profile.ndim == 1:
                                    df = pd.DataFrame({f"Posición ({unidad})": distances, "Intensidad": profile})
                                else:
                                    df = pd.DataFrame({
                                        f"Posición {unidad}": distances,
                                        "Intensidad R": profile[:, 0],
                                        "Intensidad G": profile[:, 1],
                                        "Intensidad B": profile[:, 2]
                                    })
                                    df["Promedio"] = df[["Intensidad R", "Intensidad G", "Intensidad B"]].mean(axis=1)

                                if "CurvaFiltradaPerfil" not in st.session_state:
                                    st.session_state["CurvaFiltradaPerfil"] = None

                                # Defino la columna a usar para calcular min, max y media según la opción
                                if profile.ndim == 1:
                                    if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                        columna = st.session_state["CurvaFiltradaPerfil"]
                                    else:
                                        columna = "Intensidad"
                                else:
                                    if opcion_canal == "Rojo":
                                        columna = "Intensidad R"
                                    elif opcion_canal == "Verde":
                                        columna = "Intensidad G"
                                    elif opcion_canal == "Azul":
                                        columna = "Intensidad B"
                                    elif opcion_canal == "Promedio":
                                        columna = "Promedio"
                                    elif opcion_canal == "Todos":
                                        # Para "Todos", podés usar el rango global para el slider o decidir qué hacer
                                        columna = None
                                    if opcion_canal != "Todos":
                                        if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                            columna = st.session_state["CurvaFiltradaPerfil"]

                                if columna is not None:
                                    if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                        min_val = float(columna.min())
                                        max_val = float(columna.max())
                                        media_val = float(columna.mean())
                                    else:
                                        min_val = float(df[columna].min())
                                        max_val = float(df[columna].max())
                                        media_val = float(df[columna].mean())
                                else:
                                    # Rango global entre todos los canales
                                    min_val = float(df[["Intensidad R", "Intensidad G", "Intensidad B"]].min().min())
                                    max_val = float(df[["Intensidad R", "Intensidad G", "Intensidad B"]].max().max())
                                    media_val = float(df[["Intensidad R", "Intensidad G", "Intensidad B"]].mean().mean())

                                # Slider con valores adaptados
                                if min_val < max_val:
                                    y1_int, y2_int = st.slider(
                                        "Seleccioná dos niveles de intensidad",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=(min_val, max_val),
                                        step=(max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01,
                                        format="%.4f"
                                    )
                                    delta_intensidad = abs(y2_int - y1_int)
                                else:
                                    st.error("La diferencia de intencidad es nula a lo largo del perfil")
                                    st.stop()
                                #st.info(f"🟦 Diferencia de intensidad: **{delta_intensidad:.4f}**")

                                x = distances  # Distancia a lo largo del perfil
                                if profile.ndim == 1:
                                    if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                        y = columna
                                    else:
                                        y = profile
                                else:
                                    if opcion_canal == "Rojo":
                                        y = df["Intensidad R"].values
                                    elif opcion_canal == "Verde":
                                        y = df["Intensidad G"].values
                                    elif opcion_canal == "Azul":
                                        y = df["Intensidad B"].values
                                    elif opcion_canal == "Promedio":
                                        y = df["Promedio"].values
                                    elif opcion_canal == "Todos":
                                        y = None  # si querés manejar el caso de 'Todos' distinto
                                    if opcion_canal != "Todos":
                                        if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                            y = columna
                                # 1. Filtrar los índices dentro del rectángulo
                                if y is not None and x is not None and None not in (x_ini, x_fin, y1_int, y2_int):
                                    indices_en_rectangulo = np.where(
                                        (x >= x_ini) & (x <= x_fin) & (y >= y1_int) & (y <= y2_int)
                                    )[0]
                                else:
                                    indices_en_rectangulo = None

                                # 2. Si hay puntos dentro del rectángulo, buscar máximos/mínimos
                                if indices_en_rectangulo is not None:
                                    if len(indices_en_rectangulo) > 0:
                                        x_rect = x[indices_en_rectangulo]
                                        y_rect = y[indices_en_rectangulo]

                                        idx_max = np.argmax((y_rect))
                                        idx_min = np.argmin((y_rect))

                                        x_max_abs = x_rect[idx_max]
                                        y_max_abs = y_rect[idx_max]

                                        x_min_abs = x_rect[idx_min]
                                        y_min_abs = y_rect[idx_min]

                                        fwhm, x_left, x_right, half_max = fn.calcular_fwhm(x_rect, y_rect)
                                    else:
                                        x_max_abs = y_max_abs = x_min_abs = y_min_abs = None
                                        fwhm = None
                                else:
                                    x_max_abs = y_max_abs = x_min_abs = y_min_abs = None
                                    fwhm = None
                                
                                


                                with col1:
                                    
                                    fig2, ax2 = plt.subplots()

                                    if profile.ndim == 2 and profile.shape[1] == 3:
                                        if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                            ax2.plot(distances,columna,label="Perfil de intensidad", color="black")
                                        else:
                                            if opcion_canal == "Todos":
                                                ax2.plot(distances, df["Intensidad R"], label="Rojo", color="red")
                                                ax2.plot(distances, df["Intensidad G"], label="Verde", color="green")
                                                ax2.plot(distances, df["Intensidad B"], label="Azul", color="blue")
                                            else:
                                                # Plot del canal o promedio seleccionado
                                                if columna == "Intensidad R":
                                                    color = "red"
                                                elif columna == "Intensidad G":
                                                    color = "green"
                                                elif columna == "Intensidad B":
                                                    color = "blue"
                                                elif columna == "Promedio":
                                                    color = "black"
                                                else:
                                                    color = "gray"  # o algún color por defecto

                                                ax2.plot(distances, df[columna], label=columna, color=color)

                                    else:
                                        # Escala de grises o promedio 1D
                                        if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                            ax2.plot(distances,columna,label="Perfil de intensidad", color="black")
                                        else:
                                            ax2.plot(distances, df["Intensidad"], label="Perfil de intensidad", color="black")

                                    ax2.set_title("Perfil de intensidad")
                                    ax2.set_xlabel(f"Posición ({unidad})")
                                    ax2.set_ylabel("Valor de píxel")
                                    ax2.grid()
                                    ax2.axvline(x_ini, color="cyan", linestyle="--")
                                    ax2.axvline(x_fin, color="cyan", linestyle="--")
                                    ax2.axhline(y1_int, color="orange", linestyle="--")
                                    ax2.axhline(y2_int, color="orange", linestyle="--")
                                    y_centro = (y1_int + y2_int) / 2
                                    ax2.axhline(y_centro, color="orange", linestyle=":", linewidth=1)
                                    ax2.text(distances[-1], y_centro, "Media", va="center", ha="right", color="orange", fontsize=8)
                                    if x_max_abs is not None:
                                        #ax.plot(x_min, y_min, 'gv', label='Mín.')
                                        #ax.plot(x_max, y_max, 'r^', label='Máx.')
                                        ax2.plot(x_max_abs, y_max_abs, 'r^', label='Máximo abs.')
                                        ax2.plot(x_min_abs, y_min_abs, 'gv', label='Mínimo abs.')
                                        #ax2.legend()
                                    # 📘 Mostrar leyenda si corresponde
                                    #if opcion_canal != "Promedio" or (profile.ndim == 2 and profile.shape[1] == 3):
                                    #    ax2.legend()

                                    # 📷 Mostrar figura en Streamlit
                                    fn.mostrar_figura_como_imagen(fig2)
                                    # 📁 Guardar figura en un buffer
                                    buf = io.BytesIO()
                                    fig2.savefig(buf, format="png", bbox_inches="tight")
                                    buf.seek(0)

                                    # ⬇️ Mostrar botón de descarga
                                    st.download_button(
                                        label="📥 Descargar gráfico como PNG",
                                        data=buf,
                                        file_name="perfil_intensidad.png",
                                        mime="image/png"
                                    )
                                    # Crear resumen
                                    #idx_max = int(np.argmax(profile.mean(axis=1))) if profile.ndim == 2 else int(np.argmax(profile))
                                    #idx_min = int(np.argmin(profile.mean(axis=1))) if profile.ndim == 2 else int(np.argmin(profile))
                                    distancia_t=abs(eje_max - eje_min)
                                    resumen = {
                                        "Máx Intensidad": f"{max_val:.4f}",
                                        "Min Intensidad": f"{min_val:.4f}",
                                        f"Distancia máx - mín {unidad}": f"{distancia_t:.2f}",
                                        "Diferencia entre y2 e y1": f"{delta_intensidad:.4f}",
                                        f"Distancia marcada (x_fin - x_ini) {unidad}": f"{delta:.2f}",
                                        "Media del perfil": f"{media_val:.4f}", 
                                    }
                                    if x_max_abs is not None:
                                        resumen["Máximo absoluto en área marcada (y)"] = f"{y_max_abs:.4f}"
                                        resumen[f"Máximo absoluto en área marcada (x) {unidad}"] = f"{x_max_abs:.4f}"
                                        resumen["Mínimo absoluto en área marcada (y)"] = f"{y_min_abs:.4f}"
                                        resumen[f"Mínimo absoluto en área marcada (x) {unidad}"] = f"{x_min_abs:.4f}"
                                    if fwhm is not None:
                                        resumen[f"FWHM {unidad}:"] = f"{fwhm:.2f}" 

                                    # Mostrar resumen en Streamlit
                                    st.write("📊 Resumen de intensidades")
                                    for k, v in resumen.items():
                                        st.write(f"- **{k}**: {v}")

                                    # Exportar a Excel
                                    buffer = io.BytesIO()
                                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                                        df.to_excel(writer, sheet_name="Perfil", index=False)

                                        # Para exportar el resumen, primero aplanamos listas si es necesario
                                        resumen_export = {}
                                        for k, v in resumen.items():
                                            if isinstance(v, list):
                                                for i, val in enumerate(v):
                                                    resumen_export[f"{k} [{['R', 'G', 'B'][i]}]"] = float(val)
                                            else:
                                                resumen_export[k] = float(v)

                                        resumen_df = pd.DataFrame.from_dict(resumen_export, orient="index", columns=["Valor"])
                                        resumen_df.to_excel(writer, sheet_name="Resumen")

                                    st.download_button(
                                        label="📥 Descargar perfil y resumen (Excel)",
                                        data=buffer.getvalue(),
                                        file_name="perfil_intensidad.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                with col2:
                                    fig, ax = plt.subplots()
                                    ax.imshow(img_np, cmap="gray")
                                    ax.plot([x1_orig, x2_orig], [y1_orig, y2_orig], color='red', linewidth=1)
                                    if centro is not None:
                                        ax.plot(centro[1], centro[0], "co", label="Centro")
                                    ax.set_title("Imagen original con línea de perfil")
                                    fn.mostrar_figura_como_imagen(fig)

                            if columna is not None:
                                if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                    f = UnivariateSpline(distances,st.session_state["CurvaFiltradaPerfil"],s=0)
                                else:
                                    f = UnivariateSpline(distances, df[columna], s=0)
                                tab1, tab2, tab3 = st.tabs(["📐 Integración", "🔍 Resolver f(x) = p", "🔉 Transformada de Fourier"])
                                with tab1:
                                    st.subheader("Área bajo la curva (integración)")
                                    col1, col2 = st.columns([0.3, 0.7])
                                    with col1:
                                        a_int = st.number_input("Límite inferior (a)", value=float(distances[0]), format="%.2f")
                                        b_int = st.number_input("Límite superior (b)", value=float(distances[-1]), format="%.2f")
                                        paso_defecto = max((distances[-1] - distances[0]) / len(distances), 1e-4)
                                        h_int = st.number_input("Paso h", value=paso_defecto, format="%.4f", min_value=1e-4)
                                        #h_int = st.number_input("Paso h", value=(distances[-1] - distances[0]) / len(distances), format="%.4f", min_value=1e-4)
                                        metodo = st.selectbox("Método de integración", ["simpson", "trapecio", "riemann"])
                                    with col2:
                                        if h_int <= abs(b_int - a_int):
                                            area = fn.integral(f, a_int, b_int, h=h_int, tipo=metodo)
                                            st.latex(rf"\int_{{{a_int}}}^{{{b_int}}} f(x)\,dx \approx {area:.4f}")
                                            fig, ax = plt.subplots()
                                            if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                                ax.plot(distances,columna,label="Perfil de intensidad",color="blue")
                                            else:
                                                ax.plot(distances, df[columna], label="Perfil de intensidad", color="blue")
                                            x_fill = np.linspace(a_int, b_int, 500)
                                            y_fill = f(x_fill)
                                            ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4)
                                            ax.set_xlabel("Posición (píxeles o mm)")
                                            ax.set_ylabel("Intensidad")
                                            ax.set_title("Área bajo el perfil")
                                            ax.legend()
                                            st.pyplot(fig)
                                        else:
                                            st.warning("El paso h es mayor que el intervalo.")

                                with tab2:
                                    st.subheader("Resolver ecuación f(x) = p")
                                    col1, col2 = st.columns([0.3, 0.7])
                                    with col1:
                                        if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                            p_val = st.number_input("Valor p", value=float(columna.mean()), step=0.1)
                                        else:
                                            p_val = st.number_input("Valor p", value=float(df[columna].mean()), step=0.1)
                                        a_bis = st.number_input("Límite inferior", value=float(distances[0]), format="%.2f")
                                        b_bis = st.number_input("Límite superior", value=float(distances[-1]), format="%.2f")
                                        tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
                                        max_iter = st.number_input("Máx. iteraciones", value=20, step=1)

                                    def g(x): 
                                        return f(x) - p_val

                                    # Derivada numérica simple (puede mejorar si querés)
                                    def dg(x, h=1e-5):
                                        return (g(x + h) - g(x - h)) / (2 * h)

                                    try:
                                        x_sol, _, _ = fn.biner2(g, dg, a_bis, b_bis, max_iter, tol)
                                        st.success(f"✅ f(x) ≈ {p_val} → x ≈ {x_sol:.4f}")

                                        fig, ax = plt.subplots()
                                        if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                            ax.plot(distances,columna,label="Perfil de intensidad",color="blue")
                                        else:
                                            ax.plot(distances, df[columna], label="Perfil de intensidad", color="blue")
                                        ax.axhline(p_val, color='gray', linestyle='--', label=f"p = {p_val}")
                                        ax.plot(x_sol, f(x_sol), 'go', label=f"x ≈ {x_sol:.2f}")
                                        ax.legend()
                                        with col2:
                                            st.pyplot(fig)
                                    except Exception as e:
                                        st.error(f"❌ Error: {e}")

                                with tab3:
                                    # === 1. Interpolación con spline ===
                                    if usar_curva_filtrada_perfil and st.session_state["CurvaFiltradaPerfil"] is not None:
                                        spline = UnivariateSpline(distances,columna,s=0)
                                    else:
                                        spline = UnivariateSpline(distances, df[columna], s=0)
                                    M = len(distances)  # sobremuestreo para buena resolución
                                    x_interp = np.linspace(distances[0], distances[-1], M)
                                    y_interp = spline(x_interp)

                                    # === 2. Zero-padding antes de FFT ===
                                    N_fft = 2**14  # gran potencia de 2 para buena resolución
                                    y_padded = np.zeros(N_fft)
                                    y_padded[:M] = y_interp  # función interpolada + ceros

                                    dt = (distances.max() - distances.min()) / (M - 1)  # paso espacial
                                    fm = 1 / dt
                                    nyquist = fm / 2

                                    # === 3. FFT física (sin normalizar) ===
                                    fft_vals = np.fft.fft(y_padded)
                                    fft_vals_shifted = np.fft.fftshift(fft_vals)
                                    EM = np.abs(fft_vals_shifted)

                                    # Frecuencia angular en rad/unidad física (ej: rad/mm)
                                    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))
                                    freqs_rad = 2 * np.pi * freqs

                                    # === 4. Rango interactivo ===
                                    st.markdown("### Análisis de Fourier")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        f_ini = st.number_input("Frecuencia inicial", value=float(freqs_rad.min()), format="%.4f")
                                    with col2:
                                        f_fin = st.number_input("Frecuencia final", value=float(freqs_rad.max()), format="%.4f")

                                    if f_ini<f_fin:
                                        filtro = (freqs_rad >= f_ini) & (freqs_rad <= f_fin)
                                    else:
                                        st.warning("La frecuencia inicial debe ser menor a la final")
                                        st.stop()

                                    # Gráfico magnitud
                                    fig_mag, ax_mag = plt.subplots()
                                    ax_mag.plot(freqs_rad[filtro], EM[filtro], color='blue')
                                    ax_mag.set_title("Espectro de Magnitud (centrado)")
                                    ax_mag.set_xlabel("Frecuencia angular (rad/unidad)")
                                    ax_mag.set_ylabel("Magnitud")
                                    ax_mag.grid(True)
                                    with col1:
                                        st.pyplot(fig_mag)

                                    # Gráfico de fase
                                    
                                    fase = np.angle(fft_vals_shifted)
                                    fig_fase, ax_fase = plt.subplots()
                                    markerline, stemlines, baseline = ax_fase.stem(freqs_rad[filtro], fase[filtro], basefmt=" ")
                                    plt.setp(markerline, color='orange', marker='o')
                                    plt.setp(stemlines, color='orange')
                                    ax_fase.set_title("Espectro de Fase (centrado y discreto)")
                                    ax_fase.set_xlabel("Frecuencia angular (rad/unidad)")
                                    ax_fase.set_ylabel("Fase (rad)")
                                    ax_fase.grid(True)
                                    with col2:
                                        st.pyplot(fig_fase)
                                    st.markdown("### 🔎 Comparación del espectro de magnitud (FFT centrada)")
                                    dol1, dol2 = st.columns(2)
                                    sol1, sol2 = st.columns(2)
                                    # --- Selección tipo de filtro ---
                                    with dol1:
                                        tipo_filtro = st.radio("Tipo de filtro:", ["Pasa bajos", "Pasa altos", "Pasa banda"])

                                    
                                        # --- Selección de frecuencias de corte ---
                                        if tipo_filtro == "Pasa bajos":
                                            f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.01, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                            mascara = np.abs(freqs_rad) <= f_corte

                                        elif tipo_filtro == "Pasa altos":
                                            f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.01, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                            mascara = np.abs(freqs_rad) >= f_corte

                                        elif tipo_filtro == "Pasa banda":
                                            f1 = st.number_input("Frecuencia mínima (rad/s)", min_value=0.0, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                            f2 = st.number_input("Frecuencia máxima (rad/s)", min_value=f1, max_value=nyquist*2*np.pi, value=nyquist*2*np.pi, step=0.1)
                                            mascara = (np.abs(freqs_rad) >= f1) & (np.abs(freqs_rad) <= f2)

                                        # --- Aplicar filtro al espectro ---
                                        fft_filtrada_shifted = fft_vals_shifted * mascara
                                        fft_filtrada = np.fft.ifftshift(fft_filtrada_shifted)  # volver al orden original
                                        y_filtrada = np.fft.ifft(fft_filtrada)  # antitransformada

                                        # --- Magnitud del espectro filtrado vs original ---
                                        
                                    with dol2:
                                        fig_mag_comp, ax_mag_comp = plt.subplots()
                                        ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_vals_shifted[filtro]), '--', label="Original", color='gray', alpha=0.5)
                                        ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_filtrada_shifted[filtro]), label="Filtrada", color='blue')
                                        ax_mag_comp.set_title("Magnitud del espectro: original vs filtrado")
                                        ax_mag_comp.set_xlabel("Frecuencia angular (rad/s)")
                                        ax_mag_comp.set_ylabel("Magnitud")
                                        ax_mag_comp.grid(True)
                                        ax_mag_comp.legend()
                                        st.pyplot(fig_mag_comp)

                                    # --- Visualizar reconstrucción ---
                                    st.markdown("### 🔁 Señal filtrada (dominio del espacio)")
                                    rol1,rol2=st.columns(2)

                                    mostrar_comparacion = st.checkbox("Comparar con señal original", value=True)
                                    guardar_curva_filtrada_perfil = st.button("Guardar Curva Filtrada", key="GCP")
                                    if guardar_curva_filtrada_perfil:
                                        st.session_state["CurvaFiltradaPerfil"] = np.real(y_filtrada[:M])

                                    fig_filtrada, axf = plt.subplots()
                                    if mostrar_comparacion:
                                        axf.plot(x_interp, y_interp, '--', label="Original", color='gray')
                                    axf.plot(x_interp, np.real(y_filtrada[:M]), label="Filtrada", color='blue')
                                    axf.set_xlabel(f"Posición ({unidad})")
                                    axf.set_ylabel("Intensidad")
                                    axf.set_title("Reconstrucción filtrada")
                                    axf.grid(True)
                                    axf.legend()
                                    with rol1:
                                        st.pyplot(fig_filtrada)

                                    # --- Mostrar fase filtrada opcionalmente ---
                                    fase_filtrada = np.angle(fft_filtrada_shifted)
                                    fig_fase_filt, ax_fase_filt = plt.subplots()
                                    markerline_filt, stemlines_filt, baseline_filt = ax_fase_filt.stem(freqs_rad[filtro], fase_filtrada[filtro], basefmt=" ")
                                    plt.setp(markerline_filt, color='purple', marker='o')
                                    plt.setp(stemlines_filt, color='purple')
                                    ax_fase_filt.set_title("Espectro de Fase Filtrado (centrado y discreto)")
                                    ax_fase_filt.set_xlabel("Frecuencia angular (rad/s)")
                                    ax_fase_filt.set_ylabel("Fase (rad)")
                                    ax_fase_filt.grid(True)
                                    with rol2:
                                        st.pyplot(fig_fase_filt)

                    case "ROIs":
                        st.subheader("ROI 1 y ROI 2")

                        # Preparar imagen base para mostrar
                        img_np_view = img_np.copy()
                        vmin, vmax = np.min(img_np_view), np.max(img_np_view)
                        img_view = np.clip((img_np_view - vmin) / (vmax - vmin), 0, 1)
                        img_view = (img_view * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_view).convert("RGB")

                        height_orig, width_orig = img_np.shape[:2]
                        max_canvas_size = 512
                        scale = min(max_canvas_size / width_orig, max_canvas_size / height_orig)
                        canvas_width = int(width_orig * scale)
                        canvas_height = int(height_orig * scale)
                        img_resized = img_pil.resize((canvas_width, canvas_height))

                        # Radio para elegir modo de trabajo: dibujar ROIs o ver curva ROI vs tiempo
                        #modo = st.radio("Seleccioná acción", ["Dibujar ROIs"], horizontal=True)
                        modotabs = st.tabs(["Dibujar ROIs", "Curva ROI vs. tiempo"])
                        # Selección de tipo de ROI y ROI actual a dibujar
                        with modotabs[0]:
                            roi_tipo = st.selectbox("Tipo de ROI", ["rect", "circle", "freedraw"], index=0, format_func=lambda x: {
                                "rect": "Rectángulo",
                                "circle": "Círculo",
                                "freedraw": "Mano alzada"
                            }[x])
                            roi_actual = st.radio("¿Qué ROI estás dibujando ahora?", ["ROI 1", "ROI 2"], horizontal=True)
                            color = "lime" if roi_actual == "ROI 1" else "yellow"

                            # Inicializar estado para guardar datos del canvas
                            if "canvas_data_compartido" not in st.session_state:
                                st.session_state["canvas_data_compartido"] = None

                            # Mostrar canvas
                            canvas_result = st_canvas(
                                fill_color="rgba(255,255,255,0.0)",
                                stroke_width=2,
                                stroke_color=color,
                                background_image=img_resized,
                                height=canvas_height,
                                width=canvas_width,
                                drawing_mode=roi_tipo,
                                key="canvas_compartido",
                                update_streamlit=True
                            )

                            # Guardar datos del canvas en session_state
                            if canvas_result.json_data:
                                st.session_state["canvas_data_compartido"] = canvas_result.json_data

                            roi1_objs = []
                            roi2_objs = []

                            canvas_data = st.session_state["canvas_data_compartido"]
                            if canvas_data and "objects" in canvas_data:
                                for obj in canvas_data["objects"]:
                                    stroke = obj.get("stroke", "").lower()
                                    if stroke == "lime":
                                        roi1_objs.append(obj)
                                    elif stroke == "yellow":
                                        roi2_objs.append(obj)
                            #st.write("Canvas data objects:", len(canvas_data["objects"]))
                            #for obj in canvas_data["objects"]:
                            #    if obj["type"] == "path":
                            #        st.write(obj["path"])
                            # Procesar objetos por ROI
                            roi1, patch1, stats1, mask1, info1 = fn.procesar_lista_de_objetos(roi1_objs, scale, img_np, color="lime")
                            roi2, patch2, stats2, mask2, info2 = fn.procesar_lista_de_objetos(roi2_objs, scale, img_np, color="yellow")

                            # Guardar máscaras e info para usar en curva ROI vs tiempo
                            st.session_state["mask1"] = mask1
                            st.session_state["mask2"] = mask2
                            st.session_state["info1"] = info1
                            st.session_state["info2"] = info2

                            mm_per_pixel = st.session_state.get("escala_mm_por_pixel", 1.0)

                            col1, col2 = st.columns(2)

                            with col1:
                                if roi1 is not None:
                                    mostrar_mm1 = st.checkbox("Mostrar área ROI 1 en mm²", value=False, key="mm1")
                                    area1 = stats1["area"] * mm_per_pixel**2 if mostrar_mm1 else stats1["area"]
                                    unidad1 = "mm²" if mostrar_mm1 else "píxeles"
                                    st.image(fn.normalizar_0_255(roi1), caption="ROI 1", clamp=True, use_container_width=True)
                                    st.write(f"🔢 Intensidad media ROI 1: {stats1['media']:.2f}")
                                    st.write(f"➕ Suma intensidades ROI 1: {stats1['suma']}")
                                    st.write(f"📏 Área ROI 1: {area1:.2f} {unidad1}")
                                else:
                                    st.info("Dibuja la ROI 1 para ver estadísticas.")

                            with col2:
                                if roi2 is not None:
                                    mostrar_mm2 = st.checkbox("Mostrar área ROI 2 en mm²", value=False, key="mm2")
                                    area2 = stats2["area"] * mm_per_pixel**2 if mostrar_mm2 else stats2["area"]
                                    unidad2 = "mm²" if mostrar_mm2 else "píxeles"
                                    st.image(fn.normalizar_0_255(roi2), caption="ROI 2", clamp=True, use_container_width=True)
                                    st.write(f"🔢 Intensidad media ROI 2: {stats2['media']:.2f}")
                                    st.write(f"➕ Suma intensidades ROI 2: {stats2['suma']}")
                                    st.write(f"📏 Área ROI 2: {area2:.2f} {unidad2}")
                                else:
                                    st.info("Dibuja la ROI 2 para ver estadísticas.")

                            if roi1 is not None and roi2 is not None:
                                st.markdown("### Comparación")
                                st.write(f"🔻 Diferencia suma intensidades: {abs(stats1['suma'] - stats2['suma'])}")
                                st.write(f"🔻 Diferencia media intensidades: {abs(stats1['media'] - stats2['media']):.2f}")

                            #st.markdown("### 🖼️ Imagen con ROIs superpuestas")
                            #fig, ax = plt.subplots()
                            #ax.imshow(img_np, cmap="gray")
                            #if patch1:
                            #    ax.add_patch(patch1)
                            #if patch2:
                            #    ax.add_patch(patch2)
                            #if patch1 or patch2:
                            #    ax.legend()
                            #st.pyplot(fig)

                            operacion = st.selectbox(
                                "Operación con ROIs",
                                [
                                    "Fusionar ROI 1 y ROI 2",
                                    "Intersecar ROI 1 y ROI 2",
                                    "ROI 1 sin ROI 2",
                                    "ROI 2 sin ROI 1",
                                    "Imagen sin ROI 1",
                                    "Imagen sin ROI 2",
                                    "Imagen sin ROI 1 ni ROI 2"
                                ]
                            )

                            resultado = img_np.copy()

                            if operacion == "Fusionar ROI 1 y ROI 2":
                                mascara = np.logical_or(mask1, mask2)
                                if mascara is not None and isinstance(mascara, np.ndarray) and mascara.dtype == bool:
                                    resultado[~mascara] = 0
                                caption = "Fusión ROI 1 y ROI 2"

                            elif operacion == "Intersecar ROI 1 y ROI 2":
                                mascara = np.logical_and(mask1, mask2)
                                if mascara is not None and isinstance(mascara, np.ndarray) and mascara.dtype == bool:
                                    resultado[~mascara] = 0
                                caption = "Intersección ROI 1 y ROI 2"

                            elif operacion == "ROI 1 sin ROI 2":
                                mascara = np.logical_and(mask1, ~mask2)
                                if mascara is not None and isinstance(mascara, np.ndarray) and mascara.dtype == bool:
                                    resultado[~mascara] = 0
                                caption = "ROI 1 sin ROI 2"

                            elif operacion == "ROI 2 sin ROI 1":
                                mascara = np.logical_and(mask2, ~mask1)
                                if mascara is not None and isinstance(mascara, np.ndarray) and mascara.dtype == bool:
                                    resultado[~mascara] = 0
                                caption = "ROI 2 sin ROI 1"

                            elif operacion == "Imagen sin ROI 1":
                                resultado[mask1] = 0
                                caption = "Imagen sin ROI 1"

                            elif operacion == "Imagen sin ROI 2":
                                resultado[mask2] = 0
                                caption = "Imagen sin ROI 2"

                            elif operacion == "Imagen sin ROI 1 ni ROI 2":
                                resultado[np.logical_or(mask1, mask2)] = 0
                                caption = "Imagen sin ROI 1 ni ROI 2"

                            st.image(fn.normalizar_0_255(resultado), caption=caption, use_container_width=True)

                        #subirarchivo = st.radio("Modo de subir archivo", ["Automático", "Manual (solo DICOM)"],horizontal=True )

                        with modotabs[1]:
                            
                            if subirarchivo == "Manual (solo DICOM)":                 
                                # Cargar máscaras e info de session_state
                                mask1 = st.session_state.get("mask1")
                                mask2 = st.session_state.get("mask2")
                                info1 = st.session_state.get("info1")
                                info2 = st.session_state.get("info2")

                                st.subheader("📈 ROI vs. tiempo")

                                if proy is not None and mask1 is not None and mask2 is not None:
                                    import pandas as pd
                                    from io import BytesIO

                                    N_frames = proy.shape[2]

                                    # Tiempo por frame ingresado por el usuario
                                    tiempo_por_frame = st.number_input("⏱️ Tiempo por frame (segundos)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
                                    tiempos = [i * tiempo_por_frame for i in range(N_frames)]

                                    idx_frame = st.slider("Frame para visualizar", 0, N_frames - 1, 0)
                                    woli1, woli2 = st.columns(2)
                                    frame = proy[:, :, idx_frame].astype(np.float32)

                                    # Mostrar frame con ROIs
                                    fig, ax = plt.subplots()
                                    ax.imshow(frame, cmap="gray")
                                    if info1:
                                        ax.add_patch(fn.reconstruir_patch_desde_info(info1))
                                    if info2:
                                        ax.add_patch(fn.reconstruir_patch_desde_info(info2))
                                    ax.set_title(f"Frame {idx_frame} con ROIs")
                                    ax.axis("off")
                                    if info1 or info2:
                                        ax.legend()
                                    with woli1:
                                        st.pyplot(fig)

                                    # Checkbox para mostrar área en mm²
                                    mm_per_pixel = st.session_state.get("escala_mm_por_pixel", 1.0)
                                    mostrar_mm = st.checkbox("Mostrar áreas en mm²", key="mm_curva")

                                    # Estadísticas del frame
                                    vals1 = frame[mask1]
                                    vals2 = frame[mask2]

                                    area_pix1 = np.sum(mask1)
                                    area_pix2 = np.sum(mask2)

                                    area1 = area_pix1 * mm_per_pixel**2 if mostrar_mm else area_pix1
                                    area2 = area_pix2 * mm_per_pixel**2 if mostrar_mm else area_pix2
                                    unidad_area = "mm²" if mostrar_mm else "píxeles"

                                    media1 = np.mean(vals1)
                                    suma1 = np.sum(vals1)

                                    media2 = np.mean(vals2)
                                    suma2 = np.sum(vals2)

                                    # Mostrar stats y diferencias
                                    st.markdown(f"### 📊 Estadísticas en frame {idx_frame}")

                                    col1f, col2f = st.columns(2)
                                    with col1f:
                                        st.write("🔲 ROI 1 (verde)")
                                        st.write(f"🔢 Intensidad media: {media1:.2f}")
                                        st.write(f"➕ Suma intensidades: {suma1:.2f}")
                                        st.write(f"📏 Área: {area1:.2f} {unidad_area}")
                                    with col2f:
                                        st.write("🔲 ROI 2 (amarilla)")
                                        st.write(f"🔢 Intensidad media: {media2:.2f}")
                                        st.write(f"➕ Suma intensidades: {suma2:.2f}")
                                        st.write(f"📏 Área: {area2:.2f} {unidad_area}")

                                    st.markdown("### 🔻 Diferencias")
                                    st.write(f"🔻 Diferencia suma intensidades: {abs(suma1 - suma2):.2f}")
                                    st.write(f"🔻 Diferencia media intensidades: {abs(media1 - media2):.2f}")

                                    # Calcular curvas de intensidad promedio
                                    intensidad_roi1 = [np.mean(proy[:, :, i][mask1]) for i in range(N_frames)]
                                    intensidad_roi2 = [np.mean(proy[:, :, i][mask2]) for i in range(N_frames)]

                                    f1 = UnivariateSpline(tiempos, intensidad_roi1, s=0)
                                    f2 = UnivariateSpline(tiempos, intensidad_roi2, s=0)
                                    x_full = np.linspace(tiempos[0], tiempos[-1], 1000)

                                    # Mostrar curva
                                    fig2, ax2 = plt.subplots()
                                    ax2.plot(x_full, f1(x_full), label="ROI 1", color="lime")
                                    ax2.plot(x_full, f2(x_full), label="ROI 2", color="yellow")
                                    ax2.set_xlabel("Tiempo (s)")
                                    ax2.set_ylabel("Intensidad media")
                                    ax2.set_title("Curva ROI vs. Tiempo")
                                    ax2.legend()
                                    with woli2:
                                        st.pyplot(fig2)
                                    
                                    # --- Análisis detallado spline ROI vs tiempo ---
                                    st.markdown("## 📊 Análisis de curvas")

                                    spline_opciones = st.radio("Elegí el ROI para analizar:", ["ROI 1 (verde)", "ROI 2 (amarilla)"])

                                    if "CurvaFiltradaROI" not in st.session_state:
                                        st.session_state["CurvaFiltradaROI"] = None
                                    if st.session_state["CurvaFiltradaROI"] is not None:
                                        f3 = UnivariateSpline(st.session_state["CurvaFiltradaROI"][0], st.session_state["CurvaFiltradaROI"][1],s=0)
                                    usar_curva_filtrada_roi = st.checkbox("Usar curva filtrada",key="UCFR2")

                                    f = f1 if "1" in spline_opciones else f2
                                    datos = intensidad_roi1 if "1" in spline_opciones else intensidad_roi2

                                    if usar_curva_filtrada_roi and st.session_state["CurvaFiltradaROI"] is not None:
                                        f = f3

                                    # Evaluar f en todo el dominio
                                    y_full = f(x_full)

                                    x_min_full, x_max_full = float(min(x_full)), float(max(x_full))
                                    y_min_full, y_max_full = float(min(y_full)), float(max(y_full))

                                    colcurv1, colcurv2 = st.columns([0.3, 0.7])
                                    with colcurv1:
                                        x_lines = st.slider("Intervalo en X (líneas verticales)",
                                                            min_value=x_min_full, max_value=x_max_full,
                                                            value=(x_min_full, x_max_full),
                                                            step=(x_max_full - x_min_full) / 100, format="%.2f")

                                        y_lines = st.slider("Intervalo en Y (líneas horizontales)",
                                                            min_value=y_min_full, max_value=y_max_full,
                                                            value=(y_min_full, y_max_full),
                                                            step=(y_max_full - y_min_full) / 100, format="%.2f")

                                    # Buscar máximos y mínimos dentro del rectángulo
                                    y_max = -np.inf
                                    x_max = None
                                    y_min = np.inf
                                    x_min = None

                                    for i in range(len(x_full)):
                                        x = x_full[i]
                                        y = y_full[i]
                                        if x_lines[0] <= x <= x_lines[1] and y_lines[0] <= y <= y_lines[1]:
                                            if y > y_max:
                                                y_max = y
                                                x_max = x
                                            if y < y_min:
                                                y_min = y
                                                x_min = x

                                    st.success(f"📈 Dentro del rectángulo:\n\n🔽 Mínimo f(x) = {y_min:.2f} en x = {x_min:.2f}\n🔼 Máximo f(x) = {y_max:.2f} en x = {x_max:.2f}")

                                    with colcurv2:
                                        fig_detalle, ax = plt.subplots()
                                        ax.plot(x_full, y_full, label="Spline", color='blue')
                                        ax.plot(x_min, y_min, 'gv', label='Mín.')
                                        ax.plot(x_max, y_max, 'r^', label='Máx.')

                                        ax.axvline(x=x_lines[0], color='cyan', linestyle='--', label=f'x₁ = {x_lines[0]:.2f}')
                                        ax.axvline(x=x_lines[1], color='cyan', linestyle='--', label=f'x₂ = {x_lines[1]:.2f}')
                                        ax.axhline(y=y_lines[0], color='y', linestyle='--', label=f'y₁ = {y_lines[0]:.2f}')
                                        ax.axhline(y=y_lines[1], color='y', linestyle='--', label=f'y₂ = {y_lines[1]:.2f}')
                                        ax.legend()
                                        ax.set_xlabel("Tiempo (s)")
                                        ax.set_ylabel("Intensidad")
                                        st.pyplot(fig_detalle)

                                    # --- Análisis avanzado ---
                                    tab1, tab2, tab3 = st.tabs(["📐 Integración", "🔍 Resolver f(x) = p", "🔉 Transformada de Fourier"])

                                    with tab1:
                                        st.subheader("Área bajo la curva (integración)")
                                        kol1int, kol2int = st.columns([0.3,0.7])
                                        with kol1int:
                                            a_int = st.number_input("Límite inferior (a)", value=float(tiempos[0]), format="%.2f")
                                            b_int = st.number_input("Límite superior (b)", value=float(tiempos[-1]), format="%.2f")
                                            h_int = st.number_input("Paso h", value=tiempo_por_frame, format="%.4f", min_value=1e-4)
                                            metodo = st.selectbox("Método de integración", ["simpson", "trapecio", "riemann"])
                                        if h_int <= abs(b_int - a_int):
                                            area_spline = fn.integral(f, a_int, b_int, h=h_int, tipo=metodo)
                                            with kol2int:
                                                st.latex(rf"\int_{{{a_int}}}^{{{b_int}}} f(x)\,dx \approx {area_spline:.4f}")
                                                # --- Mostrar gráfico con área sombreada ---
                                                fig_area, ax = plt.subplots()
                                                ax.plot(x_full, y_full, label="Spline", color="blue")
                                                x_fill = np.linspace(a_int, b_int, 500)
                                                y_fill = f(x_fill)
                                                ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label="Área bajo f(x)")
                                                ax.set_xlabel("Tiempo (s)")
                                                ax.set_ylabel("Intensidad")
                                                ax.set_title("Área bajo la curva")
                                                ax.legend()
                                                st.pyplot(fig_area)
                                        else:
                                            with kol2int:
                                                st.warning("El paso h es mayor que el intervalo.")

                                    with tab2:
                                        st.subheader("Resolver ecuación f(x) = p")
                                        col1re,col2re = st.columns([0.3,0.7])
                                        with col1re:
                                            p_val = st.number_input("Valor p", value=0.0, step=0.1)
                                            a_bis = st.number_input("Límite inferior", value=float(tiempos[0]), format="%.2f")
                                            b_bis = st.number_input("Límite superior", value=float(tiempos[-1]), format="%.2f")
                                            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
                                            max_iter = st.number_input("Máx. iteraciones", value=20, step=1)

                                        def g(x): return f(x) - p_val
                                        dg = f.derivative()

                                        try:
                                            x_sol, _, _ = fn.biner2(g, dg, a_bis, b_bis, max_iter, tol)
                                            st.success(f"✅ f(x) ≈ {p_val} → x ≈ {x_sol:.4f}")

                                            fig_eq, ax = plt.subplots()
                                            ax.plot(x_full, y_full, label="Spline", color='blue')
                                            ax.axhline(p_val, color='gray', linestyle='--', label=f"p = {p_val}")
                                            ax.plot(x_sol, f(x_sol), 'go', label=f"x ≈ {x_sol:.2f}")
                                            ax.legend()
                                            with col2re:
                                                st.pyplot(fig_eq)
                                        except Exception as e:
                                            st.error(f"❌ Error: {e}")
                                    with tab3:
                                        # === 1. Interpolación spline + sobremuestreo ===
                                        if usar_curva_filtrada_roi and st.session_state["CurvaFiltradaROI"] is not None:
                                            spline = f3
                                        else:
                                            spline = UnivariateSpline(tiempos, datos, s=0)
                                        M = 2048  # cantidad de puntos para interpolar
                                        x_interp = np.linspace(tiempos[0], tiempos[-1], M)
                                        y_interp = spline(x_interp)

                                        # === 2. Zero-padding antes de la FFT ===
                                        N_fft = 2**14  # potencia de 2 para buena resolución espectral
                                        y_padded = np.zeros(N_fft)
                                        y_padded[:M] = y_interp

                                        # === 3. Parámetros temporales ===
                                        dt = (max(tiempos) - min(tiempos)) / (M - 1)
                                        fm = 1 / dt
                                        nyquist = fm / 2

                                        # === 4. FFT física centrada ===
                                        fft_vals = np.fft.fft(y_padded)
                                        fft_vals_shifted = np.fft.fftshift(fft_vals)
                                        EM = np.abs(fft_vals_shifted)

                                        # === 5. Frecuencia angular real ===
                                        freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))  # Hz
                                        freqs_rad = 2 * np.pi * freqs  # rad/s

                                        # === 6. Selección de rango interactivo ===
                                        st.markdown("### Análisis de Fourier")
                                        colf1, colf2 = st.columns(2)
                                        with colf1:
                                            f_ini = st.number_input("Frecuencia inicial", value=float(freqs_rad.min()), format="%.2f")
                                        with colf2:
                                            f_fin = st.number_input("Frecuencia final", value=float(freqs_rad.max()), format="%.2f")

                                        if f_ini<f_fin:
                                            filtro = (freqs_rad >= f_ini) & (freqs_rad <= f_fin)
                                        else:
                                            st.warning("La frecuencia inicial debe ser menor a la final")
                                            st.stop()
                                        dol1,dol2=st.columns(2)
                                        sol1,sol2=st.columns(2)

                                        # === 7. Espectro de magnitud ===
                                        fig_mag, ax_mag = plt.subplots()
                                        ax_mag.plot(freqs_rad[filtro], EM[filtro], color='blue')
                                        ax_mag.set_title("Espectro de Magnitud (alta resolución)")
                                        ax_mag.set_xlabel("Frecuencia angular (rad/s)")
                                        ax_mag.set_ylabel("Magnitud")
                                        ax_mag.grid(True)
                                        with sol1:
                                            st.pyplot(fig_mag)

                                        # === 8. Fase (opcional) ===
                                        
                                    
                                        fase = np.angle(fft_vals_shifted)
                                        fig_fase, ax_fase = plt.subplots()
                                        markerline, stemlines, baseline = ax_fase.stem(freqs_rad[filtro], fase[filtro], basefmt=" ")
                                        plt.setp(markerline, color='orange', marker='o')
                                        plt.setp(stemlines, color='orange')
                                        ax_fase.set_title("Espectro de Fase (centrado y discreto)")
                                        ax_fase.set_xlabel("Frecuencia angular (rad/s)")
                                        ax_fase.set_ylabel("Fase (rad)")
                                        ax_fase.grid(True)
                                        with sol2:
                                            st.pyplot(fig_fase)
                                        
                                        st.markdown("### 🔎 Comparación del espectro de magnitud (FFT centrada)")
                                        dol1, dol2 = st.columns(2)
                                        # --- Selección tipo de filtro ---
                                        with dol1:
                                            tipo_filtro = st.radio("Tipo de filtro:", ["Pasa bajos", "Pasa altos", "Pasa banda"])

                                        
                                            # --- Selección de frecuencias de corte ---
                                            if tipo_filtro == "Pasa bajos":
                                                f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.01, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                                mascara = np.abs(freqs_rad) <= f_corte

                                            elif tipo_filtro == "Pasa altos":
                                                f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.01, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                                mascara = np.abs(freqs_rad) >= f_corte

                                            elif tipo_filtro == "Pasa banda":
                                                f1 = st.number_input("Frecuencia mínima (rad/s)", min_value=0.0, max_value=nyquist*2*np.pi, value=nyquist*np.pi/2, step=0.1)
                                                f2 = st.number_input("Frecuencia máxima (rad/s)", min_value=f1, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                                mascara = (np.abs(freqs_rad) >= f1) & (np.abs(freqs_rad) <= f2)

                                        # --- Aplicar filtro al espectro ---
                                        fft_filtrada_shifted = fft_vals_shifted * mascara
                                        fft_filtrada = np.fft.ifftshift(fft_filtrada_shifted)  # volver al orden original
                                        y_filtrada = np.fft.ifft(fft_filtrada)  # antitransformada

                                        # --- Magnitud del espectro filtrado vs original ---
                                        
                                        with dol2:
                                            fig_mag_comp, ax_mag_comp = plt.subplots()
                                            ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_vals_shifted[filtro]), '--', label="Original", color='gray', alpha=0.5)
                                            ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_filtrada_shifted[filtro]), label="Filtrada", color='blue')
                                            ax_mag_comp.set_title("Magnitud del espectro: original vs filtrado")
                                            ax_mag_comp.set_xlabel("Frecuencia angular (rad/s)")
                                            ax_mag_comp.set_ylabel("Magnitud")
                                            ax_mag_comp.grid(True)
                                            ax_mag_comp.legend()
                                            st.pyplot(fig_mag_comp)

                                        # --- Visualizar reconstrucción ---
                                        st.markdown("### 🔁 Señal filtrada (dominio del tiempo)")
                                        rol1,rol2=st.columns(2)

                                        mostrar_comparacion = st.checkbox("Comparar con señal original", value=True)
                                        guardar_curva_filtrada_roi = st.button("Guardar Curva Filtrada", key="GCR2")

                                        if guardar_curva_filtrada_roi:
                                            st.session_state["CurvaFiltradaROI"] = x_interp,np.real(y_filtrada[:M])

                                        fig_filtrada, axf = plt.subplots()
                                        if mostrar_comparacion:
                                            if usar_curva_filtrada_roi and st.session_state["CurvaFiltradaROI"] is not None:
                                                axf.plot(x_interp,st.session_state["CurvaFiltradaROI"][1], "--", label="Original", color="gray")
                                            else:
                                                axf.plot(x_interp, y_interp, '--', label="Original", color='gray')
                                        axf.plot(x_interp, np.real(y_filtrada[:M]), label="Filtrada", color='blue')
                                        axf.set_xlabel("Tiempo (s)")
                                        axf.set_ylabel("Intensidad")
                                        axf.set_title("Reconstrucción filtrada")
                                        axf.grid(True)
                                        axf.legend()
                                        with rol1:
                                            st.pyplot(fig_filtrada)

                                        # --- Mostrar fase filtrada opcionalmente ---
                                        fase_filtrada = np.angle(fft_filtrada_shifted)
                                        fig_fase_filt, ax_fase_filt = plt.subplots()
                                        markerline_filt, stemlines_filt, baseline_filt = ax_fase_filt.stem(freqs_rad[filtro], fase_filtrada[filtro], basefmt=" ")
                                        plt.setp(markerline_filt, color='purple', marker='o')
                                        plt.setp(stemlines_filt, color='purple')
                                        ax_fase_filt.set_title("Espectro de Fase Filtrado (centrado y discreto)")
                                        ax_fase_filt.set_xlabel("Frecuencia angular (rad/s)")
                                        ax_fase_filt.set_ylabel("Fase (rad)")
                                        ax_fase_filt.grid(True)
                                        with rol2:
                                            st.pyplot(fig_fase_filt)



                                    # Crear único DataFrame con todo junto
                                    df = pd.DataFrame({
                                        "Tiempo (s)": tiempos,
                                        "Intensidad ROI 1": intensidad_roi1,
                                        "Intensidad ROI 2": intensidad_roi2
                                    })

                                    # Crear Excel en memoria
                                    output_excel = BytesIO()
                                    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
                                        df.to_excel(writer, sheet_name="Curvas ROI", index=False)
                                    output_excel.seek(0)

                                    # Botón de descarga
                                    st.download_button(
                                        label="📥 Descargar curvas como Excel",
                                        data=output_excel,
                                        file_name="curvas_ROI.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                else:
                                    st.warning("Dibujá ambas ROIs y cargá un archivo multiframe para generar la curva.")  
                            elif subirarchivo == "Automático":
                                # Cargar máscaras e info de session_state
                                mask1 = st.session_state.get("mask1")
                                mask2 = st.session_state.get("mask2")
                                info1 = st.session_state.get("info1")
                                info2 = st.session_state.get("info2")

                                st.subheader("📈 ROI vs. tiempo")
                                proy = st.session_state.get("proy")
                                if proy is not None and mask1 is not None and mask2 is not None:

                                    # === BLOQUE AUTOMÁTICO ROI vs TIEMPO ===
                                    N_frames = proy.shape[0]  # N_frames está en eje 0

                                    # Tiempo por frame
                                    tiempo_por_frame = st.number_input("⏱️ Tiempo por frame (segundos)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
                                    usartiempo = st.checkbox("Usar tiempo")
                                    if usartiempo:
                                        tiempos = [i * tiempo_por_frame for i in range(N_frames)]
                                        unidad = "Tiempo [s]"
                                    else:
                                        tiempos = [i * 1 for i in range(N_frames)]
                                        unidad = "Frames"
                                    

                                    # Selección de frame actual
                                    idx_frame = st.slider("Frame para visualizar", 0, N_frames - 1, 0)
                                    woli1, woli2 = st.columns(2)

                                    frame = proy[idx_frame, :, :].astype(np.float32)

                                    # Mostrar frame con ROIs
                                    fig, ax = plt.subplots()
                                    ax.imshow(frame, cmap="gray")
                                    if info1:
                                        ax.add_patch(fn.reconstruir_patch_desde_info(info1))
                                    if info2:
                                        ax.add_patch(fn.reconstruir_patch_desde_info(info2))
                                    ax.set_title(f"Frame {idx_frame} con ROIs")
                                    ax.axis("off")
                                    if info1 or info2:
                                        ax.legend()
                                    with woli1:
                                        st.pyplot(fig)

                                    # Escala
                                    mm_per_pixel = st.session_state.get("escala_mm_por_pixel", 1.0)
                                    mostrar_mm = st.checkbox("Mostrar áreas en mm²", key="mm_curva")

                                    # Asegurar que las máscaras coincidan en tamaño
                                    shape_frame = frame.shape
                                    if mask1.shape != shape_frame:
                                        mask1_resized = resize(mask1.astype(float), shape_frame, order=0, preserve_range=True).astype(bool)
                                    else:
                                        mask1_resized = mask1

                                    if mask2.shape != shape_frame:
                                        mask2_resized = resize(mask2.astype(float), shape_frame, order=0, preserve_range=True).astype(bool)
                                    else:
                                        mask2_resized = mask2

                                    # Estadísticas
                                    vals1 = frame[mask1_resized]
                                    vals2 = frame[mask2_resized]

                                    area_pix1 = np.sum(mask1_resized)
                                    area_pix2 = np.sum(mask2_resized)

                                    area1 = area_pix1 * mm_per_pixel**2 if mostrar_mm else area_pix1
                                    area2 = area_pix2 * mm_per_pixel**2 if mostrar_mm else area_pix2
                                    unidad_area = "mm²" if mostrar_mm else "píxeles"

                                    media1 = np.mean(vals1)
                                    suma1 = np.sum(vals1)

                                    media2 = np.mean(vals2)
                                    suma2 = np.sum(vals2)

                                    st.markdown(f"### 📊 Estadísticas en frame {idx_frame}")
                                    col1f, col2f = st.columns(2)
                                    with col1f:
                                        st.write("🔲 ROI 1 (verde)")
                                        st.write(f"🔢 Intensidad media: {media1:.2f}")
                                        st.write(f"➕ Suma intensidades: {suma1:.2f}")
                                        st.write(f"📏 Área: {area1:.2f} {unidad_area}")
                                    with col2f:
                                        st.write("🔲 ROI 2 (amarilla)")
                                        st.write(f"🔢 Intensidad media: {media2:.2f}")
                                        st.write(f"➕ Suma intensidades: {suma2:.2f}")
                                        st.write(f"📏 Área: {area2:.2f} {unidad_area}")

                                    st.markdown("### 🔻 Diferencias")
                                    st.write(f"🔻 Diferencia suma intensidades: {abs(suma1 - suma2):.2f}")
                                    st.write(f"🔻 Diferencia media intensidades: {abs(media1 - media2):.2f}")

                                    # Curvas
                                    intensidad_roi1 = [np.mean(proy[i, :, :][mask1_resized]) for i in range(N_frames)]
                                    intensidad_roi2 = [np.mean(proy[i, :, :][mask2_resized]) for i in range(N_frames)]

                                    fig2, ax2 = plt.subplots()
                                    ax2.plot(tiempos, intensidad_roi1, label="ROI 1", color="lime")
                                    ax2.plot(tiempos, intensidad_roi2, label="ROI 2", color="yellow")
                                    ax2.set_xlabel(f"{unidad}")
                                    ax2.set_ylabel("Intensidad media")
                                    ax2.set_title("Curva ROI vs. Tiempo")
                                    ax2.legend()
                                    with woli2:
                                        st.pyplot(fig2)
                                

                                    # --- Análisis detallado spline ROI vs tiempo ---
                                    st.markdown("## 📊 Análisis de curvas")

                                    spline_opciones = st.radio("Elegí el ROI para analizar:", ["ROI 1 (verde)", "ROI 2 (amarilla)"])
                                    f1 = UnivariateSpline(tiempos, intensidad_roi1, s=0)
                                    f2 = UnivariateSpline(tiempos, intensidad_roi2, s=0)
                                    if "CurvaFiltradaROI" not in st.session_state:
                                        st.session_state["CurvaFiltradaROI"] = None
                                    if st.session_state["CurvaFiltradaROI"] is not None:
                                        f3 = UnivariateSpline(st.session_state["CurvaFiltradaROI"][0], st.session_state["CurvaFiltradaROI"][1],s=0)
                                    usar_curva_filtrada_roi = st.checkbox("Usar curva filtrada",key="UCFR")

                                    f = f1 if "1" in spline_opciones else f2
                                    datos = intensidad_roi1 if "1" in spline_opciones else intensidad_roi2
                                    
                                    if usar_curva_filtrada_roi and st.session_state["CurvaFiltradaROI"] is not None:
                                        f = f3

                                    # Evaluar f en todo el dominio
                                    x_full = np.linspace(tiempos[0], tiempos[-1], 1000)
                                    y_full = f(x_full)

                                    x_min_full, x_max_full = float(min(x_full)), float(max(x_full))
                                    y_min_full, y_max_full = float(min(y_full)), float(max(y_full))

                                    colcurv1, colcurv2 = st.columns([0.3, 0.7])
                                    with colcurv1:
                                        x_lines = st.slider("Intervalo en X (líneas verticales)",
                                                            min_value=x_min_full, max_value=x_max_full,
                                                            value=(x_min_full, x_max_full),
                                                            step=(x_max_full - x_min_full) / 100, format="%.2f")

                                        y_lines = st.slider("Intervalo en Y (líneas horizontales)",
                                                            min_value=y_min_full, max_value=y_max_full,
                                                            value=(y_min_full, y_max_full),
                                                            step=(y_max_full - y_min_full) / 100, format="%.2f")

                                    # Buscar máximos y mínimos dentro del rectángulo
                                    y_max = -np.inf
                                    x_max = None
                                    y_min = np.inf
                                    x_min = None

                                    for i in range(len(x_full)):
                                        x = x_full[i]
                                        y = y_full[i]
                                        if x_lines[0] <= x <= x_lines[1] and y_lines[0] <= y <= y_lines[1]:
                                            if y > y_max:
                                                y_max = y
                                                x_max = x
                                            if y < y_min:
                                                y_min = y
                                                x_min = x

                                    st.success(f"📈 Dentro del rectángulo:\n\n🔽 Mínimo f(x) = {y_min:.2f} en x = {x_min:.2f}\n🔼 Máximo f(x) = {y_max:.2f} en x = {x_max:.2f}")

                                    with colcurv2:
                                        fig_detalle, ax = plt.subplots()
                                        ax.plot(x_full, y_full, label="Spline", color='blue')
                                        ax.plot(x_min, y_min, 'gv', label='Mín.')
                                        ax.plot(x_max, y_max, 'r^', label='Máx.')

                                        ax.axvline(x=x_lines[0], color='cyan', linestyle='--', label=f'x₁ = {x_lines[0]:.2f}')
                                        ax.axvline(x=x_lines[1], color='cyan', linestyle='--', label=f'x₂ = {x_lines[1]:.2f}')
                                        ax.axhline(y=y_lines[0], color='y', linestyle='--', label=f'y₁ = {y_lines[0]:.2f}')
                                        ax.axhline(y=y_lines[1], color='y', linestyle='--', label=f'y₂ = {y_lines[1]:.2f}')
                                        ax.legend()
                                        ax.set_xlabel(f"{unidad}")
                                        ax.set_ylabel("Intensidad")
                                        st.pyplot(fig_detalle)

                                    # --- Análisis avanzado ---
                                    tab1,tab2,tab3 = st.tabs(["📐 Integración", "🔍 Resolver f(x) = p", "🔉 Transformada de Fourier"])

                                    with tab1:
                                        st.subheader("Área bajo la curva (integración)")
                                        kol1int, kol2int = st.columns([0.3,0.7])
                                        with kol1int:
                                            a_int = st.number_input("Límite inferior (a)", value=float(tiempos[0]), format="%.2f")
                                            b_int = st.number_input("Límite superior (b)", value=float(tiempos[-1]), format="%.2f")
                                            h_int = st.number_input("Paso h", value=tiempo_por_frame, format="%.4f", min_value=1e-4)
                                            metodo = st.selectbox("Método de integración", ["simpson", "trapecio", "riemann"])
                                        if h_int <= abs(b_int - a_int):
                                            area_spline = fn.integral(f, a_int, b_int, h=h_int, tipo=metodo)
                                            with kol2int:
                                                st.latex(rf"\int_{{{a_int}}}^{{{b_int}}} f(x)\,dx \approx {area_spline:.4f}")
                                                # --- Mostrar gráfico con área sombreada ---
                                                fig_area, ax = plt.subplots()
                                                ax.plot(x_full, y_full, label="Spline", color="blue")
                                                x_fill = np.linspace(a_int, b_int, 500)
                                                y_fill = f(x_fill)
                                                ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.4, label="Área bajo f(x)")
                                                ax.set_xlabel(f"{unidad}")
                                                ax.set_ylabel("Intensidad")
                                                ax.set_title("Área bajo la curva")
                                                ax.legend()
                                                st.pyplot(fig_area)
                                        else:
                                            with kol2int:
                                                st.warning("El paso h es mayor que el intervalo.")

                                    with tab2:
                                        st.subheader("Resolver ecuación f(x) = p")
                                        col1re,col2re = st.columns([0.3,0.7])
                                        with col1re:
                                            p_val = st.number_input("Valor p", value=0.0, step=0.1)
                                            a_bis = st.number_input("Límite inferior", value=float(tiempos[0]), format="%.2f")
                                            b_bis = st.number_input("Límite superior", value=float(tiempos[-1]), format="%.2f")
                                            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
                                            max_iter = st.number_input("Máx. iteraciones", value=20, step=1)

                                        def g(x): return f(x) - p_val
                                        dg = f.derivative()

                                        try:
                                            x_sol, _, _ = fn.biner2(g, dg, a_bis, b_bis, max_iter, tol)
                                            st.success(f"✅ f(x) ≈ {p_val} → x ≈ {x_sol:.4f}")

                                            fig_eq, ax = plt.subplots()
                                            ax.plot(x_full, y_full, label="Spline", color='blue')
                                            ax.axhline(p_val, color='gray', linestyle='--', label=f"p = {p_val}")
                                            ax.plot(x_sol, f(x_sol), 'go', label=f"x ≈ {x_sol:.2f}")
                                            ax.legend()
                                            with col2re:
                                                st.pyplot(fig_eq)
                                        except Exception as e:
                                            st.error(f"❌ Error: {e}")
                                    
                                        with tab3:
                                            # === 1. Interpolación spline + sobremuestreo ===
                                            if usar_curva_filtrada_roi and st.session_state["CurvaFiltradaROI"] is not None:
                                                spline = f3
                                            else:
                                                spline = UnivariateSpline(tiempos, datos, s=0)
                                            M = 2048  # cantidad de puntos para interpolar
                                            x_interp = np.linspace(tiempos[0], tiempos[-1], M)
                                            y_interp = spline(x_interp)

                                            # === 2. Zero-padding antes de la FFT ===
                                            N_fft = 2**14  # potencia de 2 para buena resolución espectral
                                            y_padded = np.zeros(N_fft)
                                            y_padded[:M] = y_interp

                                            # === 3. Parámetros temporales ===
                                            dt = (max(tiempos) - min(tiempos)) / (M - 1)
                                            fm = 1 / dt
                                            nyquist = fm / 2

                                            # === 4. FFT física centrada ===
                                            fft_vals = np.fft.fft(y_padded)
                                            fft_vals_shifted = np.fft.fftshift(fft_vals)
                                            EM = np.abs(fft_vals_shifted)

                                            # === 5. Frecuencia angular real ===
                                            freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))  # Hz
                                            freqs_rad = 2 * np.pi * freqs  # rad/s

                                            # === 6. Selección de rango interactivo ===
                                            st.markdown("### Análisis de Fourier")
                                            colf1, colf2 = st.columns(2)
                                            with colf1:
                                                f_ini = st.number_input("Frecuencia inicial", value=float(freqs_rad.min()), format="%.2f")
                                            with colf2:
                                                f_fin = st.number_input("Frecuencia final", value=float(freqs_rad.max()), format="%.2f")

                                            if f_ini<f_fin:
                                                filtro = (freqs_rad >= f_ini) & (freqs_rad <= f_fin)
                                            else:
                                                st.warning("La frecuencia inicial debe ser menor a la final")
                                                st.stop()
                                            dol1,dol2=st.columns(2)
                                            sol1,sol2=st.columns(2)

                                            # === 7. Espectro de magnitud ===
                                            fig_mag, ax_mag = plt.subplots()
                                            ax_mag.plot(freqs_rad[filtro], EM[filtro], color='blue')
                                            ax_mag.set_title("Espectro de Magnitud (alta resolución)")
                                            ax_mag.set_xlabel("Frecuencia angular (rad/s)")
                                            ax_mag.set_ylabel("Magnitud")
                                            ax_mag.grid(True)
                                            with sol1:
                                                st.pyplot(fig_mag)

                                            # === 8. Fase (opcional) ===
                                            
                                        
                                            fase = np.angle(fft_vals_shifted)
                                            fig_fase, ax_fase = plt.subplots()
                                            markerline, stemlines, baseline = ax_fase.stem(freqs_rad[filtro], fase[filtro], basefmt=" ")
                                            plt.setp(markerline, color='orange', marker='o')
                                            plt.setp(stemlines, color='orange')
                                            ax_fase.set_title("Espectro de Fase (centrado y discreto)")
                                            ax_fase.set_xlabel("Frecuencia angular (rad/s)")
                                            ax_fase.set_ylabel("Fase (rad)")
                                            ax_fase.grid(True)
                                            with sol2:
                                                st.pyplot(fig_fase)
                                            
                                            st.markdown("### 🔎 Comparación del espectro de magnitud (FFT centrada)")
                                            dol1, dol2 = st.columns(2)
                                            # --- Selección tipo de filtro ---
                                            with dol1:
                                                tipo_filtro = st.radio("Tipo de filtro:", ["Pasa bajos", "Pasa altos", "Pasa banda"])

                                            
                                                # --- Selección de frecuencias de corte ---
                                                if tipo_filtro == "Pasa bajos":
                                                    f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.01, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                                    mascara = np.abs(freqs_rad) <= f_corte

                                                elif tipo_filtro == "Pasa altos":
                                                    f_corte = st.number_input("Frecuencia de corte (rad/s)", min_value=0.01, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                                    mascara = np.abs(freqs_rad) >= f_corte

                                                elif tipo_filtro == "Pasa banda":
                                                    f1 = st.number_input("Frecuencia mínima (rad/s)", min_value=0.0, max_value=nyquist*2*np.pi, value=nyquist*np.pi/2, step=0.1)
                                                    f2 = st.number_input("Frecuencia máxima (rad/s)", min_value=f1, max_value=nyquist*2*np.pi, value=nyquist*np.pi, step=0.1)
                                                    mascara = (np.abs(freqs_rad) >= f1) & (np.abs(freqs_rad) <= f2)

                                            # --- Aplicar filtro al espectro ---
                                            fft_filtrada_shifted = fft_vals_shifted * mascara
                                            fft_filtrada = np.fft.ifftshift(fft_filtrada_shifted)  # volver al orden original
                                            y_filtrada = np.fft.ifft(fft_filtrada)  # antitransformada

                                            # --- Magnitud del espectro filtrado vs original ---
                                            
                                            with dol2:
                                                fig_mag_comp, ax_mag_comp = plt.subplots()
                                                ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_vals_shifted[filtro]), '--', label="Original", color='gray', alpha=0.5)
                                                ax_mag_comp.plot(freqs_rad[filtro], np.abs(fft_filtrada_shifted[filtro]), label="Filtrada", color='blue')
                                                ax_mag_comp.set_title("Magnitud del espectro: original vs filtrado")
                                                ax_mag_comp.set_xlabel("Frecuencia angular (rad/s)")
                                                ax_mag_comp.set_ylabel("Magnitud")
                                                ax_mag_comp.grid(True)
                                                ax_mag_comp.legend()
                                                st.pyplot(fig_mag_comp)

                                            # --- Visualizar reconstrucción ---
                                            st.markdown("### 🔁 Señal filtrada (dominio del tiempo)")
                                            rol1,rol2=st.columns(2)

                                            mostrar_comparacion = st.checkbox("Comparar con señal original", value=True)
                                            guardar_curva_filtrada_roi = st.button("Guardar Curva Filtrada", key="GCR")
                                            if guardar_curva_filtrada_roi:
                                                st.session_state["CurvaFiltradaROI"] = x_interp,np.real(y_filtrada[:M])

                                            fig_filtrada, axf = plt.subplots()
                                            if mostrar_comparacion:
                                                if usar_curva_filtrada_roi and st.session_state["CurvaFiltradaROI"] is not None:
                                                    axf.plot(x_interp,st.session_state["CurvaFiltradaROI"][1], "--", label="Original", color="gray")
                                                else:
                                                    axf.plot(x_interp, y_interp, '--', label="Original", color='gray')
                                            axf.plot(x_interp, np.real(y_filtrada[:M]), label="Filtrada", color='blue')
                                            axf.set_xlabel(f"{unidad}")
                                            axf.set_ylabel("Intensidad")
                                            axf.set_title("Reconstrucción filtrada")
                                            axf.grid(True)
                                            axf.legend()
                                            with rol1:
                                                st.pyplot(fig_filtrada)

                                            # --- Mostrar fase filtrada opcionalmente ---
                                            fase_filtrada = np.angle(fft_filtrada_shifted)
                                            fig_fase_filt, ax_fase_filt = plt.subplots()
                                            markerline_filt, stemlines_filt, baseline_filt = ax_fase_filt.stem(freqs_rad[filtro], fase_filtrada[filtro], basefmt=" ")
                                            plt.setp(markerline_filt, color='purple', marker='o')
                                            plt.setp(stemlines_filt, color='purple')
                                            ax_fase_filt.set_title("Espectro de Fase Filtrado (centrado y discreto)")
                                            ax_fase_filt.set_xlabel("Frecuencia angular (rad/s)")
                                            ax_fase_filt.set_ylabel("Fase (rad)")
                                            ax_fase_filt.grid(True)
                                            with rol2:
                                                st.pyplot(fig_fase_filt)

                                        #st.info(f"""
                                        #    La frecuencia de muestreo es {fm:.2f} Hz → frecuencia de Nyquist: {nyquist:.2f} Hz ({2*np.pi*nyquist:.2f} rad/s).
                                        #    Se usa `fftshift` para centrar el espectro como en MATLAB.
                                        #""")

                                    # Crear único DataFrame con todo junto
                                    df = pd.DataFrame({
                                        "Tiempo (s)": tiempos,
                                        "Intensidad ROI 1": intensidad_roi1,
                                        "Intensidad ROI 2": intensidad_roi2
                                    })

                                    # Crear Excel en memoria
                                    output_excel = BytesIO()
                                    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
                                        df.to_excel(writer, sheet_name="Curvas ROI", index=False)
                                    output_excel.seek(0)

                                    # Botón de descarga
                                    st.download_button(
                                        label="📥 Descargar curvas como Excel",
                                        data=output_excel,
                                        file_name="curvas_ROI.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                                    
                    case "ROIs2":
                        with tolum1:
                            # ROI 1 con centro + ancho/alto
                            tolumna1, tolumna2 = st.columns(2)
                            with tolumna1:
                                st.write("🔲 ROI rectangular 1")

                                max_cx = img_np.shape[1] - 1
                                max_cy = img_np.shape[0] - 1

                                cx_roi1 = st.number_input(
                                    "Centro x ROI 1", 
                                    min_value=0, 
                                    max_value=max_cx, 
                                    value=max_cx // 2, 
                                    key="roi1_cx"
                                )

                                cy_roi1 = st.number_input(
                                    "Centro y ROI 1", 
                                    min_value=0, 
                                    max_value=max_cy, 
                                    value=max_cy // 2, 
                                    key="roi1_cy"
                                )

                                max_w1 = min(cx_roi1 * 2, (img_np.shape[1] - cx_roi1) * 2)
                                max_h1 = min(cy_roi1 * 2, (img_np.shape[0] - cy_roi1) * 2)

                                w_roi1 = st.number_input(
                                    "Ancho ROI 1", 
                                    min_value=1, 
                                    max_value=max_w1, 
                                    value=min(50, max_w1), 
                                    key="roi1_w"
                                )

                                h_roi1 = st.number_input(
                                    "Alto ROI 1", 
                                    min_value=1, 
                                    max_value=max_h1, 
                                    value=min(50, max_h1), 
                                    key="roi1_h"
                                )

                                    # Calcular límites del ROI a partir del centro
                                x1 = max(0, int(cx_roi1 - w_roi1 / 2))
                                x2 = min(img_np.shape[1], int(cx_roi1 + w_roi1 / 2))
                                y1 = max(0, int(cy_roi1 - h_roi1 / 2))
                                y2 = min(img_np.shape[0], int(cy_roi1 + h_roi1 / 2))

                                roi1 = img_np[y1:y2, x1:x2]
                                st.image(fn.normalizar_0_255(roi1), caption="ROI 1", clamp=True, use_container_width=True)
                                st.write(f"🔢 Intensidad media ROI 1: {np.mean(roi1):.2f}")
                                st.write(f"➕ Suma de intensidades ROI 1: {np.sum(roi1)}")
                                st.write(f"📏 Área ROI 1: {roi1.size} píxeles")

                            with tolumna2:
                                # ROI 2
                                st.write("🔲 ROI rectangular 2")

                                cx_roi2 = st.number_input(
                                    "Centro x ROI 2", 
                                    min_value=0, 
                                    max_value=img_np.shape[1] - 1, 
                                    value=min(30, img_np.shape[1] - 1), 
                                    key="roi2_cx"
                                )

                                cy_roi2 = st.number_input(
                                    "Centro y ROI 2", 
                                    min_value=0, 
                                    max_value=img_np.shape[0] - 1, 
                                    value=min(30, img_np.shape[0] - 1), 
                                    key="roi2_cy"
                                )

                                w_roi2 = st.number_input(
                                    "Ancho ROI 2", 
                                    min_value=1, 
                                    max_value=min(100, img_np.shape[1]), 
                                    value=50, 
                                    key="roi2_w"
                                )

                                h_roi2 = st.number_input(
                                    "Alto ROI 2", 
                                    min_value=1, 
                                    max_value=min(100, img_np.shape[0]), 
                                    value=50, 
                                    key="roi2_h"
                                )

                                    # Calcular esquina superior izquierda
                                x_roi2 = int(max(0, cx_roi2 - w_roi2 // 2))
                                y_roi2 = int(max(0, cy_roi2 - h_roi2 // 2))

                                    # Recorte del ROI 2 (asegurando que no se pase de los límites)
                                roi2 = img_np[y_roi2:y_roi2 + h_roi2, x_roi2:x_roi2 + w_roi2]

                                st.image(fn.normalizar_0_255(roi2), caption="ROI 2", clamp=True, use_container_width=True)
                                st.write(f"🔢 Intensidad media ROI 2: {np.mean(roi2):.2f}")
                                st.write(f"➕ Suma de intensidades ROI 1: {np.sum(roi2)}")
                                st.write(f"📏 Área ROI 2: {roi2.size} píxeles")
                            st.write(f" Diferencia de intensidad total: {abs(np.sum(roi1)-np.sum(roi2))}")
                            st.write(f" Diferencia de intensidad media: {abs(np.mean(roi1)-np.mean(roi2)):.2f}")

                        with tolum2:
                                # Mostrar imagen original con línea y ROIs
                            st.subheader("🖼️ Imagen con superposición")
                            fig, ax = plt.subplots()
                            ax.imshow(img_np, cmap="gray")

                                # Calcular esquinas superiores izquierdas a partir del centro
                            x1_roi1 = cx_roi1 - w_roi1 / 2
                            y1_roi1 = cy_roi1 - h_roi1 / 2
                            x1_roi2 = cx_roi2 - w_roi2 / 2
                            y1_roi2 = cy_roi2 - h_roi2 / 2

                            rect1 = plt.Rectangle((x1_roi1, y1_roi1), w_roi1, h_roi1, 
                                                edgecolor='lime', facecolor='none', linewidth=1.5, label='ROI 1')
                            rect2 = plt.Rectangle((x1_roi2, y1_roi2), w_roi2, h_roi2, 
                                                edgecolor='yellow', facecolor='none', linewidth=1.5, label='ROI 2')

                            ax.add_patch(rect1)
                            ax.add_patch(rect2)
                            ax.set_title("Imagen con ROIs")
                            ax.legend()
                            st.pyplot(fig)
    case "Sobre mí":
        st.title("Sobre mí")
        st.markdown("""
        Mi nombre es **David Fernández Basti**. Soy **Técnico en Diagnóstico por Imágenes** por la Universidad Nacional de San Martín (**UNSAM**) y estudiante de **Ingeniería Nuclear**.

        Desarrollé esta aplicación como un proyecto personal para integrar herramientas que suelo usar en el ámbito académico: desde álgebra lineal hasta procesamiento de imágenes médicas y reconstrucción tomográfica.

        Mi objetivo es que esta herramienta sea útil tanto para estudiantes de tecnicaturas como ingenierías.

        Podés contactarme en:
        - [LinkedIn](https://www.linkedin.com/in/david-fern%C3%A1ndez-basti-133114190/)
                    

        ---
        ### 💙 ¿Te gustó la app?
        Apoyá el proyecto con una donación:
        [📲 Donar vía MercadoPago](http://link.mercadopago.com.ar/datrix)
         Más información en:
        [⭐ Dejame una estrella](https://github.com/David-Basti/datrix/stargazers)
        """)
