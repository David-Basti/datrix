# Funciones
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from skimage.color import rgb2gray
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage import img_as_float
from PIL import Image
import Funciones as fn
from scipy.ndimage import zoom
from math import ceil, sqrt
import io
import tempfile
import matplotlib.animation as animation
import os
import scipy.ndimage
from matplotlib.animation import writers
from matplotlib.animation import PillowWriter,FFMpegWriter
from scipy.ndimage import gaussian_filter
import imageio
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import base64
from io import BytesIO
from matplotlib.patches import Rectangle, Ellipse, Polygon
from scipy.interpolate import interp1d
import pandas as pd

# Funcio iradon
def iradon_with_spline(getp, theta, output_size):
    # Genera la reconstrucción usando iradon (sin interpolación avanzada)
    O = iradon(getp, theta=theta, filter_name=None, circle=False, output_size=output_size, interpolation='linear')
    
    # Ahora aplicar interpolación tipo spline usando scipy
    O_spline = zoom(O, 2, order=3)  # 'order=3' aplica un filtro cúbico (similar a spline)
    return O_spline

#Normalizar
def normalizar(im):
    im_min = np.min(im)
    im_max = np.max(im)
    if im_max - im_min == 0:
        return np.zeros_like(im)
    return (im - im_min) / (im_max - im_min)

# Función para texto personalizado
def titulo_personalizado(texto, nivel=1, tamaño=36, color='black', centrado=True):
    alineacion = 'center' if centrado else 'left'
    html = f"<h{nivel} style='text-align: {alineacion}; font-size: {tamaño}px; color: {color};'>{texto}</h{nivel}>"
    st.markdown(html, unsafe_allow_html=True)

# Función para convertir imagen en matriz
def image_to_matrix(img):
    #img = img.resize((n, m))  # Escala de grises
    return np.array(img)

# Producto bin a bin
def mult_elementwise(A, B):
    if A.shape != B.shape:
        raise ValueError("Las matrices deben tener la misma dimensión.")
    return sp.Matrix([[A[i, j] * B[i, j] for j in range(A.cols)] for i in range(A.rows)])

# Division bin a bin
def divide_elementwise(A, B):
    if A.shape != B.shape:
        raise ValueError("Las matrices deben tener la misma dimensión.")
    return sp.Matrix([[A[i, j] / B[i, j] for j in range(A.cols)] for i in range(A.rows)])

#prox producto matricial
def prox(v, w):
    # Caso especial: vector columna por vector fila → escalar
    if v.shape[1] == 1 and w.shape[0] == 1 and v.shape[0] == w.shape[1]:
        return (w * v)[0]  # Producto escalar (número)

    # Producto matricial general
    if v.shape[1] != w.shape[0]:
        raise ValueError("Número de columnas de v debe ser igual al número de filas de w")

    r = sp.zeros(v.shape[0], w.shape[1])
    for m in range(v.shape[0]):
        for n in range(w.shape[1]):
            b = 0
            for k in range(v.shape[1]):
                b += v[m, k] * w[k, n]
            r[m, n] = b
    return r
#def ajustar_a_128(img):
    # Convertimos a float si no lo es
 #   if img.dtype != np.float32 and img.dtype != np.float64:
  #      img = img_as_float(img)

    # Redimensionamos forzando a 128x128
   # img_redimensionada = resize(img, (128, 128), anti_aliasing=True, preserve_range=True)
    
    #return img_redimensionada
#def ajustar_a_128(img):
    # Verificamos las dimensiones originales de la imagen
#    original_shape = img.shape
#    max_dim = max(original_shape)

    # Redimensionar manteniendo proporciones para que el lado más largo sea 128
 #   escala = 128 / max_dim
 #   nueva_shape = tuple([int(dim * escala) for dim in original_shape])
 #   img_redimensionada = resize(img, nueva_shape, anti_aliasing=True)

    # Calcular el padding necesario para llegar a 128x128
 #   pad_alto = 128 - img_redimensionada.shape[0]
 #   pad_ancho = 128 - img_redimensionada.shape[1]
    
 #   pad_arriba = pad_alto // 2
 #   pad_abajo = pad_alto - pad_arriba
  #  pad_izquierda = pad_ancho // 2
   # pad_derecha = pad_ancho - pad_izquierda

    # Aplicar padding
   # img_final = np.pad(img_redimensionada, ((pad_arriba, pad_abajo), (pad_izquierda, pad_derecha)), mode='constant', constant_values=0)

   # return img_final

# Función para entrada manual
def parse_manual_input(texto):
    return np.array([[float(num) for num in fila.split()] for fila in texto.strip().split('\n')])
    
# Función para visualizar imagen
def mostrar_matriz(matriz, titulo="Matriz"):
    fig, ax = plt.subplots()
    ax.imshow(matriz, cmap='gray', interpolation='nearest')
    ax.set_title(titulo)
    ax.axis('off')
    st.pyplot(fig)

# Expandir imagen
def expandir_a_fov(imagen):
    alto, ancho = imagen.shape
    nuevo_ancho = int(ceil(sqrt(2) * max(alto, ancho)))
    nueva_imagen = np.zeros((nuevo_ancho, nuevo_ancho), dtype=imagen.dtype)

    # Centrar la imagen original en la nueva matriz
    offset_x = (nuevo_ancho - ancho) // 2
    offset_y = (nuevo_ancho - alto) // 2
    nueva_imagen[offset_y:offset_y + alto, offset_x:offset_x + ancho] = imagen

    return nueva_imagen

# Llevar \FOV a cero
def aplicar_mascara_circular(imagen,factor_radio=1):
    #imagen = imagen.astype(float)
    imagen = img_as_float(imagen)
    y, x = np.indices(imagen.shape)
    centro = np.array(imagen.shape) / 2
    radio = min(imagen.shape)*factor_radio / 2
    mascara = (x - centro[1])**2 + (y - centro[0])**2 <= radio**2
    return imagen * mascara 
    
    #if imagen.dtype == np.uint8:
        # Usamos np.where para mantener los valores 0-255 y evitar problemas con bool * uint8
    #    imagen_filtrada = np.where(mascara, imagen, 0)
    #    return imagen_filtrada.astype(np.uint8)
    #else:
        # Si es float u otro tipo, podés usar multiplicación directa
    #    return imagen * mascara

# FBP
def FBP(I, a, b, p, sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    reconstrucciones = []
    O = iradon(getp, theta=angulos, filter_name="ramp",
               interpolation="linear", circle=True, output_size=I.shape[0])
    O[O < 0] = 0
    geto = radon(O, theta=angulos, circle=True)
    return I, O, getp, geto,reconstrucciones
    #if not modo_vid:
    #    return I, O, getp, geto,reconstrucciones
    #else:
    #    for i in range(1, len(angulos)+2):
        # Se va agregando cada paso de la reconstrucción
    #        O = iradon(getp[:, :i+1], theta=angulos[:i+1], filter_name="ramp", interpolation="linear", circle=True, output_size=I.shape[0])
    #        O[O < 0] = 0
    #        reconstrucciones.append(O.copy())
    #return I, O, getp, geto, reconstrucciones

# FBP_video
def FBP_vid(I, a, b, p,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    reconstrucciones = []
    for i in range(1, len(angulos)+2):
            O = iradon(getp[:, :i+1], theta=angulos[:i+1], filter_name="ramp", interpolation="linear", circle=True, output_size=I.shape[0])
            O[O < 0] = 0
            reconstrucciones.append(O.copy())
    return reconstrucciones 

# MLEM
def MLEM(I, N, a, b, p,modo_O,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    match modo_O:
        case "Imagen en blanco":
            O = np.ones_like(I)
        case "Imagen por FBP":
            O = iradon(getp, theta=angulos, filter_name="hann",interpolation="linear", circle=True, output_size=I.shape[0])
            O[O < 0] = 0
            O = gaussian_filter(O, sigma=1)
    Ir = iradon(getp, theta=angulos, filter_name=None, interpolation='linear', circle=True, output_size=I.shape[0])
    norfO = iradon(np.ones_like(radon(O, theta=angulos)),theta=angulos, filter_name=None, interpolation='linear', circle=True, output_size=O.shape[0])
    for _ in range(N):
        est = radon(O, theta=angulos, circle=True)
        fA = np.divide(getp, est, out=np.zeros_like(getp), where=est != 0)
        fAA = iradon(fA, theta=angulos, filter_name=None, interpolation='linear', circle=True, output_size=O.shape[0])
        O = np.divide(O * fAA, norfO, out=np.zeros_like(O), where=norfO != 0)
    geto = radon(O, theta=angulos, circle=True)
    # Mostrar con Streamlit
    return I, O, getp ,geto

# MLEM_video
def MLEM_vid(I, N, a, b, p,modo_O,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    reconstrucciones = []
    match modo_O:
        case "Imagen en blanco":
            O = np.ones_like(I)
            reconstrucciones.append(O.copy())
        case "Imagen por FBP":
            O = iradon(getp, theta=angulos, filter_name="hann",interpolation="linear", circle=True, output_size=I.shape[0])
            O[O < 0] = 0
            O = gaussian_filter(O, sigma=1)
            reconstrucciones.append(O.copy())
    Ir = iradon(getp, theta=angulos, filter_name=None, interpolation='linear', circle=True, output_size=I.shape[0])
    norfO = iradon(np.ones_like(radon(O, theta=angulos)),theta=angulos, filter_name=None, interpolation='linear', circle=True, output_size=O.shape[0])
    for _ in range(N):
        est = radon(O, theta=angulos, circle=True)
        fA = np.divide(getp, est, out=np.zeros_like(getp), where=est != 0)
        fAA = iradon(fA, theta=angulos, filter_name=None, interpolation='linear', circle=True, output_size=O.shape[0])
        O = np.divide(O * fAA, norfO, out=np.zeros_like(O), where=norfO != 0)
        reconstrucciones.append(O.copy())
    return reconstrucciones

def OSEM(I, N, a, b, p, subsets, modo_O,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    num_ang = len(angulos)
    if subsets < 1 or subsets > len(angulos):
         st.error("El número de subsets debe ser mayor que 0 y menor o igual a la cantidad de ángulos.")
         st.stop()
    elif len(angulos) % subsets != 0:
         st.warning("⚠️ El número de subsets no divide exactamente a la cantidad de ángulos. La reconstrucción aún se hará, pero puede ser subóptima.")
    subset_size = num_ang // subsets
    I = img_as_float(I)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    match modo_O:
        case "Imagen en blanco":
            O = np.ones_like(I)
        case "Imagen por FBP":
            O = iradon(getp, theta=angulos, filter_name="hann",interpolation="linear", circle=True, output_size=I.shape[0])
            O[O < 0] = 0
            O = gaussian_filter(O, sigma=1)

    for i in range(N):
        for s in range(subsets):
            subset_ang = angulos[s::subsets]
            subset_getp = getp[:, s::subsets]
            est = radon(O, theta=subset_ang, circle=True)

            fA = np.divide(subset_getp, est, out=np.zeros_like(est), where=est != 0)
            fAA = iradon(fA, theta=subset_ang, filter_name=None, circle=True, output_size=I.shape[0])
            norm = iradon(np.ones_like(est), theta=subset_ang, filter_name=None, circle=True, output_size=I.shape[0])
            norm[norm<0] = 0
            O *= fAA / np.maximum(norm, 1e-10)
    geto = radon(O, theta=angulos, circle=True)
    # Mostrar con Streamlit
    return I, O, getp ,geto

def OSEM_vid(I,N,a,b,p,subsets,modo_O,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    num_ang = len(angulos)
    if subsets < 1 or subsets > len(angulos):
         st.error("El número de subsets debe ser mayor que 0 y menor o igual a la cantidad de ángulos.")
         st.stop()
    elif len(angulos) % subsets != 0:
         st.warning("⚠️ El número de subsets no divide exactamente a la cantidad de ángulos. La reconstrucción aún se hará, pero puede ser subóptima.")
    subset_size = num_ang // subsets
    I = img_as_float(I)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    reconstrucciones = []
    match modo_O:
        case "Imagen en blanco":
            O = np.ones_like(I)
            reconstrucciones.append(O.copy())
        case "Imagen por FBP":
            O = iradon(getp, theta=angulos, filter_name="hann",interpolation="linear", circle=True, output_size=I.shape[0])
            O[O < 0] = 0
            O = gaussian_filter(O, sigma=1)
            reconstrucciones.append(O.copy())

    for i in range(N):
        for s in range(subsets):
            subset_ang = angulos[s::subsets]
            subset_getp = getp[:, s::subsets]
            est = radon(O, theta=subset_ang, circle=True)

            fA = np.divide(subset_getp, est, out=np.zeros_like(est), where=est != 0)
            fAA = iradon(fA, theta=subset_ang, filter_name=None, circle=True, output_size=I.shape[0])
            norm = iradon(np.ones_like(est), theta=subset_ang, filter_name=None, circle=True, output_size=I.shape[0])
            norm[norm<0] = 0
            O *= fAA / np.maximum(norm, 1e-10)
            reconstrucciones.append(O.copy())
    return reconstrucciones

def SART(I, N, a, b, p,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)

    # Reconstrucción inicial con SART
    O = iradon_sart(getp, theta=angulos)

    # Refinamiento iterativo
    for _ in range(N - 1):
        O = iradon_sart(getp, theta=angulos, image=O)
    
    # Mostrar resultados
    geto = radon(O, theta=angulos, circle=True)

    return I, O, getp ,geto

def SART_vid(I,N,a,b,p,sinograma=None):
    I = img_as_float(I)
    angulos = np.arange(a, b, p)
    if sinograma is not None:
        getp = sinograma
    else:
        getp = radon(I, theta=angulos, circle=True)
    reconstrucciones = []
    # Reconstrucción inicial con SART
    O = iradon_sart(getp, theta=angulos)
    reconstrucciones.append(O.copy())

    # Refinamiento iterativo
    for _ in range(N - 1):
        O = iradon_sart(getp, theta=angulos, image=O)
        reconstrucciones.append(O.copy())
    
    # Mostrar resultados
    return reconstrucciones

def crear_gif_desde_multiframe(pixel_array, nombre="dicom_animado.gif", fps=5):
    # Normalizar los frames
    frames = []
    for frame in pixel_array:
        img = frame.astype(np.float32)
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype(np.uint8)
        frames.append(img)
    temp_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    
    frames_grandes = [cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST) for frame in frames]
    imageio.mimsave(temp_gif.name, frames_grandes, duration=1/fps, loop=0)
    return temp_gif.name

def crear_gif_manual(secuencia,I,nombre="dicom_animado2.gif", fps=5):
    frames = []
    for k in range(I-1):
        q = secuencia[:,:,k]
        img = q.astype(np.float32)
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype(np.uint8)
        frames.append(img)
    temp_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    frames_grandes = [cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST) for frame in frames]
    imageio.mimsave(temp_gif.name, frames_grandes, duration=1/fps, loop=0)
    return temp_gif.name


def crear_video_reconstruccion(reconstrucciones, output_path, writer="pillow", fps=5):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    img = ax.imshow(reconstrucciones[0], cmap='gray')
    ax.axis("off")  # Oculta los ejes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    #def update(i):
    #    img.set_data(reconstrucciones[i])
    #    return [img]
    def update(i):
        frame = reconstrucciones[i]
        img.set_data(frame)
        img.set_clim(vmin=frame.min(), vmax=frame.max())  # Esto es como imshow(I, []) en MATLAB
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(reconstrucciones), blit=True)
    ani.save(output_path, writer=PillowWriter(fps=fps))

# CONVETIR A HU
def convertir_a_hounsfield(imagen, mu_agua=0.167):
    """Convierte la imagen a unidades Hounsfield.

    Args:
        imagen (np.ndarray): imagen con coeficientes de atenuación.
        mu_agua (float): coeficiente de atenuación del agua (por defecto 0.2).

    Returns:
        np.ndarray: imagen en unidades Hounsfield (HU).
    """
    return 1000 * (imagen - mu_agua) / mu_agua

# VENTANAS

def aplicar_ventana(imagen, L, W):
    """Aplica ventaneo a la imagen, como en TC.

    Args:
        imagen (np.ndarray): matriz de valores de atenuación.
        L (float): nivel de ventana (center).
        W (float): ancho de ventana (width).

    Returns:
        np.ndarray: imagen con ventana aplicada (escalada de 0 a 1).
    """
    img = imagen.copy()
    min_val = L - W / 2
    max_val = L + W / 2

    # Clip y escalar
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)

    return img

# Ventaneo imágenes TC
def aplicar_ventaneo(imagen, L, W):
    img_min = L - (W / 2)
    img_max = L + (W / 2)
    ventana = np.clip(imagen, img_min, img_max)
    ventana = (ventana - img_min) / (img_max - img_min)  # Normalizar 0-1
    ventana = (ventana * 255).astype(np.uint8)           # Escalar 0-255
    return ventana
# FDISCOS

def fantoma_discos_atenuacion():
    # Crear malla
    y = np.arange(-63, 65)
    x = np.arange(-63, 65)
    u, v = np.meshgrid(x, y)

# Disco grande de agua (radio 64, centrado en 0,0)
    z_agua = np.sqrt(62.5**2 - u**2 - v**2)
    z_agua[np.isnan(z_agua)] = 0
    mask_agua = z_agua > 0

# Discos chicos de hueso
    discos = [
    np.sqrt(10**2 - (u + 34.65)**2 - (v + 34.65)**2),   # z6
    np.sqrt(9**2  - (u + 49)**2   - (v)**2),             # z1
    np.sqrt(8**2  - (u + 34.65)**2 - (v - 34.65)**2),    # z2
    np.sqrt(7**2  - (u)**2        - (v - 49)**2),        # z3
    np.sqrt(6**2  - (u - 34.65)**2 - (v - 34.65)**2),    # z4
    np.sqrt(5**2  - (u - 49)**2   - (v)**2),             # z5
    np.sqrt(13**2 - (u - 34.65)**2 - (v + 34.65)**2),    # z
    np.sqrt(12**2 - (u**2 + (v + 49)**2)),               # z0    
    ]

# Máscara de hueso: unir todos los discos chicos
    mask_hueso = np.zeros_like(u, dtype=bool)
    for d in discos:
        d[np.isnan(d)] = 0
        mask_hueso |= (d > 0)

# Inicializar imagen
    Z = np.zeros_like(u, dtype=float)
    Z[mask_agua] = 0.167    # Agua
    Z[mask_hueso] = 0.6   # Hueso (sobrescribe al agua si se solapan)

    # Suma del fondo + las esferas
    #imagen_final = np.array(Z_mask) + np.array(ZC_mask)
    #imagen_final = Z+ZC

    return Z

# Fantoma solo agua
def fantoma_agua_atenuacion():
    # Crear malla
    y = np.arange(-63, 65)
    x = np.arange(-63, 65)
    u, v = np.meshgrid(x, y)

# Disco grande de agua (radio 64, centrado en 0,0)
    z_agua = np.sqrt(62.5**2 - u**2 - v**2)
    z_agua[np.isnan(z_agua)] = 0
    mask_agua = z_agua > 0

# Inicializar imagen
    Z = np.zeros_like(u, dtype=float)
    Z[mask_agua] = 0.167    # Agua
    return Z

def fantoma_discos_actividad():
    # Crear malla
    y = np.arange(-63, 65)
    x = np.arange(-63, 65)
    u, v = np.meshgrid(x, y)

# Disco grande de agua (radio 64, centrado en 0,0)
    z_agua = np.sqrt(62.5**2 - u**2 - v**2)
    z_agua[np.isnan(z_agua)] = 0
    mask_agua = z_agua > 0

# Discos chicos de hueso
    discos = [
    #np.sqrt(10**2 - (u + 34.65)**2 - (v + 34.65)**2),   # z6
    #np.sqrt(9**2  - (u + 49)**2   - (v)**2),             # z1
    #np.sqrt(8**2  - (u + 34.65)**2 - (v - 34.65)**2),    # z2
    np.sqrt(4**2  - (u)**2 -v**2),        # z3
    #np.sqrt(6**2  - (u - 34.65)**2 - (v - 34.65)**2),    # z4
    #np.sqrt(5**2  - (u - 49)**2   - (v)**2),             # z5
    #np.sqrt(13**2 - (u - 34.65)**2 - (v + 34.65)**2),    # z
    np.sqrt(4**2 - (u + 49)**2 - (v)**2),               # z0    
    ]

# Máscara de hueso: unir todos los discos chicos
    mask_hueso = np.zeros_like(u, dtype=bool)
    for d in discos:
        d[np.isnan(d)] = 0
        mask_hueso |= (d > 0)

# Inicializar imagen
    Z = np.zeros_like(u, dtype=float)
    Z[mask_agua] = 35    # Agua
    Z[mask_hueso] = 200   # Hueso (sobrescribe al agua si se solapan)

    # Suma del fondo + las esferas
    #imagen_final = np.array(Z_mask) + np.array(ZC_mask)
    #imagen_final = Z+ZC

    return Z

# FTORAX

def fantoma_torax_atenuacion():
    y = np.arange(-127, 128)
    x = np.arange(-127, 128)
    u, v = np.meshgrid(x, y)

# Máscara de elipse externa (base del fantoma)
    mask_base = ((u**2 / 121**2) + (v**2 / 64**2)) <= 1

# Inicializar la imagen
    Z = np.zeros_like(u, dtype=float)

# ==================
# TEJIDO BLANDO (0.3)
# ==================

# Elipse central (grande)
    mask_tb1 = ((u**2 / 100**2) + (v**2 / 50**2)) <= 1

# Elipse interna (para que el tejido quede encerrado entre dos elipses)
#mask_tb2 = ((u**2 / 16**2) + (v**2 / 20**2)) <= 1
#Z[mask_tb2] = 0.3

# Asignar valor de tejido blando donde se cumplan esas condiciones y dentro del fantoma
    mask_tb_total = (mask_base | mask_tb1)# & mask_base
    Z[mask_tb_total] = 0.16

# ==========
# AIRE (0.0)
# ==========

# Dos elipses simétricas horizontales
    mask_air1 = ((u - 50)**2 / 45**2) + (v**2 / 35**2) <= 1
    mask_air2 = ((u + 50)**2 / 45**2) + (v**2 / 35**2) <= 1

# Elipse desplazada hacia abajo
    mask_air3 = (u**2 / 50**2) + ((v + 32)**2 / 16**2) <= 1

# Asignar valor de aire
    Z[(mask_air1 | mask_air2 | mask_air3) & mask_base] = 0.0

# ==========
# HUESO (0.6)
# ==========

    mask_hueso1 = (u**2 / 20**2) + ((v - 40)**2 / 20**2) <= 1
    mask_hueso2 = (u**2 / 10**2) + ((v + 50)**2 / 2**2) <= 1
    mask_hueso3 = ((u - 70)**2 / 5**2) + ((v + 40)**2 / 2**2) <= 1
    mask_hueso4 = ((u + 70)**2 / 5**2) + ((v + 40)**2 / 2**2) <= 1
    mask_hueso5 = ((u - 100)**2 / 2**2) + (v**2 / 5**2) <= 1
    mask_hueso6 = ((u + 100)**2 / 2**2) + (v**2 / 5**2) <= 1
    mask_hueso7 = (u**2 / 50**2) + ((v - 55)**2 / 5**2) <= 1  # posible error: se usó y en la ecuación, adaptado a v

# Unir todas las máscaras
    mask_hueso_total = (mask_hueso1 | mask_hueso2 | mask_hueso3 | mask_hueso4 |
                    mask_hueso5 | mask_hueso6 | mask_hueso7) & mask_base

# Asignar valor de hueso
    Z[mask_hueso_total] = 0.6

# Aplicar máscara base (todo lo que está fuera de la elipse mayor se mantiene en cero)
    Z[~mask_base] = 0.0
    mask_tb2 = ((u**2 / 16**2) + ((v+20)**2 / 20**2)) <= 1
    Z[mask_tb2] = 0.18
    Z = resize(Z, (128, 128), anti_aliasing=True, preserve_range=True)
    
    return Z

# Fan actividad
def crear_mapa_actividad(N=128, radio=15, A1=1.0, desplazamiento=40):
    # Crear grilla de coordenadas
    y, x = np.indices((N, N))
    centro = np.array([N//2, N//2])

    # Disco central
    dist_centro = np.sqrt((x - centro[1])**2 + (y - centro[0])**2)
    disco_central = dist_centro <= radio

    # Disco periférico (por ejemplo, hacia la derecha)
    centro_periferico = centro + np.array([0, desplazamiento])
    dist_periferico = np.sqrt((x - centro_periferico[1])**2 + (y - centro_periferico[0])**2)
    disco_periferico = dist_periferico <= radio

    # Inicializar mapa de actividad
    actividad = np.zeros((N, N))

    # Asignar actividad A1 a ambos discos
    actividad[disco_central] = A1
    actividad[disco_periferico] = A1

    return actividad

def siddon_ray_trace(src, det, shape, pixel_size):
    """Devuelve los índices de píxeles atravesados por el rayo y longitudes dentro de cada uno."""
    nx, ny = shape
    dx, dy = pixel_size, pixel_size
    x0, y0 = src
    x1, y1 = det

    L = np.hypot(x1 - x0, y1 - y0)
    if L == 0:
        return [], []

    # Número máximo de píxeles en x o y
    N = max(nx, ny)
    t = np.linspace(0, 1, N * 3)  # suficientemente fino
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)

    i = np.clip((x // dx).astype(int), 0, nx - 1)
    j = np.clip((y // dy).astype(int), 0, ny - 1)
    pixels = list(zip(i, j))

    unique_pixels = []
    lengths = []
    prev = None
    dt = L / len(t)

    for p in pixels:
        if p != prev:
            unique_pixels.append(p)
            lengths.append(dt)
            prev = p
        else:
            lengths[-1] += dt

    return unique_pixels, lengths

def forward_projection_simple(activity, mu, angulos, circle=True):
    """
    Simula un sinograma SPECT aproximado:
      - activity: mapa de actividad (N×N)
      - mu:       mapa de coeficientes de atenuación (N×N)
      - angulos:  array con los ángulos de proyección en grados o radianes
      - circle:   si True, asume FOV circular

    Devuelve:
      sinogram:  array shape (len(angulos), N)  
    """
    # 1) Proyección directa de la actividad
    sino_act = radon(activity, theta=angulos, circle=True)

    # 2) Proyección de las atenuaciones (line integrals de μ)
    sino_mu  = radon(mu,  theta=angulos, circle=True)

    # 3) Modelo SPECT: cada línea de actividad se atenúa exponencialmente
    sinogram = sino_act * np.exp(-sino_mu*0.2)

    return sinogram


def fantoma_aire():
    Z = np.zeros([128,128])
    return Z

# Función para aplicar filtro
def aplicar_filtro(img, kernel):
    return cv2.filter2D(img, -1, kernel)


# Función para crear máscaras
def crear_filtro(shape, tipo, fc1_rel, fc2_rel=None,orden=1):
    filas, cols = shape
    u = np.arange(-cols // 2, cols // 2)
    v = np.arange(-filas // 2, filas // 2)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)

    # Distancia máxima posible desde el centro a la esquina
    Dmax = np.sqrt((cols // 2)**2 + (filas // 2)**2)
    Dnorm = D / D.max()  # frecuencia normalizada [0, 1]

    # Escalamos las frecuencias relativas
    fc1 = fc1_rel * Dmax
    fc2 = fc2_rel * Dmax if fc2_rel is not None else None

    if tipo == "Pasa bajo":
        return D <= fc1
    elif tipo == "Pasa alto":
        return D >= fc1
    elif tipo == "Pasa banda":
        return (D >= fc1) & (D <= fc2)
    elif tipo == "Gaussiano":
        return np.exp(-(D**2) / (2 * (fc1**2)))
    elif tipo == "Pasa alto Gaussiano":
        return 1 - np.exp(-(D**2) / (2 * (fc1**2)))
    elif tipo == "Rampa":
        return Dnorm
    elif tipo == "Parzen":
        H = np.zeros_like(Dnorm)
        mask = Dnorm <= fc1_rel
        r = Dnorm[mask] / fc1_rel
        H[mask] = 1 - 6*r**2 + 6*r**3
        return H

    elif tipo == "Shepp-Logan":
        H = np.zeros_like(Dnorm)
        mask = Dnorm <= fc1_rel
        x = np.pi * Dnorm[mask] / fc1_rel
        H[mask] = np.sinc(x / np.pi)
        return H

    elif tipo == "Hann":
        H = np.zeros_like(Dnorm)
        mask = Dnorm <= fc1_rel
        H[mask] = 0.5 * (1 + np.cos(np.pi * Dnorm[mask] / fc1_rel))
        return H

    elif tipo == "Hamming":
        H = np.zeros_like(Dnorm)
        mask = Dnorm <= fc1_rel
        H[mask] = 0.54 + 0.46 * np.cos(np.pi * Dnorm[mask] / fc1_rel)
        return H
    
    elif tipo == "Butterworth":
        return 1 / (1 + (D / fc1)**(2 * orden))

    else:
        raise ValueError("Tipo de filtro no reconocido.")
    
# Función para aplicar filtro en frecuencia

def filtro_frecuencia(img, tipo, fc1, fc2=None,orden=None):
    F = fftshift(fft2(img))
    mask = crear_filtro(img.shape, tipo, fc1, fc2,orden)
    Ffiltrado = F * mask
    img_filtrada = np.abs(ifft2(ifftshift(Ffiltrado)))
    return np.log(1 + np.abs(F)), mask, np.log(1 + np.abs(Ffiltrado)), img_filtrada


def normalizar_0_255(im):
    im_norm = im - np.min(im)
    im_norm = im_norm / np.max(im_norm)
    return (im_norm * 255).astype(np.uint8)

# Función para convertir imagen a base64
def pil_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extraer_roi(obj, scale, img_np):
    left = int(obj["left"] / scale)
    top = int(obj["top"] / scale)
    width = int(obj["width"] / scale)
    height = int(obj["height"] / scale)
    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(img_np.shape[1], x1 + width)
    y2 = min(img_np.shape[0], y1 + height)
    return img_np[y1:y2, x1:x2], left, top, width, height

def extraer_puntos_path(path_data, scale):
    pts = []
    for cmd in path_data:
        tipo = cmd[0]
        if tipo == "M" or tipo == "L":
            # comando simple con 2 coords (x,y)
            x = cmd[1]
            y = cmd[2]
            pts.append([int(x / scale), int(y / scale)])
        elif tipo == "Q":
            # curva cuadrática: punto control + punto final
            # tomamos solo el punto final para aproximar
            x = cmd[3]
            y = cmd[4]
            pts.append([int(x / scale), int(y / scale)])
        else:
            # otros comandos si aparecen se pueden ignorar o manejar aquí
            pass
    return np.array(pts, dtype=np.int32)

def procesar_lista_de_objetos(objs, scale, img_np, color="lime"):
    from matplotlib.patches import Rectangle, Circle, Polygon
    import numpy as np
    import cv2
    import re

    if not objs:
        return None, None, None, None, None

    height, width = img_np.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    last_patch = None

    for obj in objs:
        shape = obj["type"]
        if shape == "rect":
            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            width_rect = int(obj["width"] / scale)
            height_rect = int(obj["height"] / scale)
            mask[top:top+height_rect, left:left+width_rect] = 1
            last_patch = Rectangle((left, top), width_rect, height_rect, edgecolor=color, fill=False)
        elif shape == "circle":
            cx = int((obj["left"] + obj["width"] / 2) / scale)
            cy = int((obj["top"] + obj["height"] / 2) / scale)
            radius = int(obj["width"] / 2 / scale)
            cv2.circle(mask, (cx, cy), radius, 1, thickness=-1)
            last_patch = Circle((cx, cy), radius, edgecolor=color, fill=False)
        elif shape == "freedraw" or shape == "path":
            path_data = obj.get("path")
            if path_data:
                pts_array = extraer_puntos_path(path_data, scale)
                if len(pts_array) >= 3:
                    cv2.fillPoly(mask, [pts_array], 1)
                    last_patch = Polygon(pts_array, edgecolor=color, fill=False)



    if not np.any(mask):
        return None, None, None, None, None

    roi = img_np.copy()
    roi[mask == 0] = 0

    stats = {
        "suma": float(np.sum(img_np[mask == 1])),
        "media": float(np.mean(img_np[mask == 1])),
        "area": int(np.sum(mask))
    }

    info = {
    "tipo": shape,
    "objeto": obj,
    "color": color,
    "label": f"ROI {color}",
    "scale": scale
    }

    return roi, last_patch, stats, mask.astype(bool), info

def obtener_roi_desde_canvas(canvas_result, scale, img_np, color="lime", nombre="ROI"):
    height_orig, width_orig = img_np.shape[:2]

    if canvas_result and "objects" in canvas_result:
        objs = canvas_result["objects"]
        if len(objs) == 0:
            return None, None, None, None, None

        obj = objs[-1]
        tipo = obj["type"]

        info_figura = {
            "tipo": tipo,
            "objeto": obj,
            "color": color,
            "label": nombre,
            "scale": scale
        }

        if tipo == "rect":
            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            width = int(obj["width"] / scale)
            height = int(obj["height"] / scale)

            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(width_orig, x1 + width)
            y2 = min(height_orig, y1 + height)

            roi = img_np[y1:y2, x1:x2]

            patch = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  edgecolor=color, facecolor='none', linewidth=1.5, label=nombre)

            mask_bin = np.zeros((height_orig, width_orig), dtype=bool)
            mask_bin[y1:y2, x1:x2] = True

            stats = {
                "media": np.mean(roi),
                "suma": np.sum(roi),
                "area": roi.shape[0] * roi.shape[1]
            }
            return roi, patch, stats, mask_bin, info_figura

        elif tipo == "circle":
            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            width = int(obj.get("width", 0) / scale)
            height = int(obj.get("height", 0) / scale)

            x1 = max(0, left)
            y1 = max(0, top)
            x2 = min(width_orig, x1 + width)
            y2 = min(height_orig, y1 + height)

            roi = img_np[y1:y2, x1:x2]

            cx = (x2 - x1) // 2
            cy = (y2 - y1) // 2
            rx = (x2 - x1) // 2
            ry = (y2 - y1) // 2

            yy, xx = np.ogrid[:y2 - y1, :x2 - x1]
            mask_local = ((xx - cx)**2) / (rx**2) + ((yy - cy)**2) / (ry**2) <= 1

            roi_masked = roi.copy()
            if roi.ndim == 2:
                roi_masked[~mask_local] = 0
            else:
                roi_masked[~mask_local] = 0

            patch = Ellipse((x1 + cx, y1 + cy), 2*rx, 2*ry,
                            edgecolor=color, facecolor='none', linewidth=1.5, label=nombre)

            mask_bin = np.zeros((height_orig, width_orig), dtype=bool)
            mask_bin[y1:y2, x1:x2] = mask_local

            pixels_roi = roi[mask_local] if roi.ndim == 2 else roi[mask_local].reshape(-1, roi.shape[2])
            stats = {
                "media": float(np.mean(pixels_roi)) if pixels_roi.size > 0 else 0,
                "suma": float(np.sum(pixels_roi)) if pixels_roi.size > 0 else 0,
                "area": int(np.sum(mask_local))
            }

            return roi_masked, patch, stats, mask_bin, info_figura

        elif tipo == "path":
            points = [(int(p[1] / scale), int(p[2] / scale)) for p in obj["path"]]
            if len(points) < 3:
                return None, None, None, None, None

            roi_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

            try:
                import cv2
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(roi_mask, [pts], 1)
            except ImportError:
                from matplotlib.path import Path
                poly_path = Path(points)
                for y in range(height_orig):
                    for x in range(width_orig):
                        if poly_path.contains_point((x, y)):
                            roi_mask[y, x] = 1

            coords = np.argwhere(roi_mask)
            if coords.size == 0:
                return None, None, None, None, None

            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)

            roi = img_np[y1:y2+1, x1:x2+1]
            mask_crop = roi_mask[y1:y2+1, x1:x2+1]

            if roi.ndim == 2:
                roi = roi[:, :, np.newaxis]
            roi_masked = roi * mask_crop[:, :, np.newaxis]

            patch = Polygon(points, closed=True, edgecolor=color, facecolor='none', linewidth=1.5, label=nombre)

            pixels_roi = roi[mask_crop == 1]
            stats = {
                "media": float(np.mean(pixels_roi)) if pixels_roi.size > 0 else 0,
                "suma": float(np.sum(pixels_roi)) if pixels_roi.size > 0 else 0,
                "area": int(np.sum(mask_crop))
            }

            if roi_masked.shape[2] == 1:
                roi_masked = roi_masked[:, :, 0]

            mask_bin = roi_mask.astype(bool)
            return roi_masked, patch, stats, mask_bin, info_figura

        else:
            return None, None, None, None, None
    else:
        return None, None, None, None, None

def reconstruir_patch_desde_info(info):
    from matplotlib.patches import Rectangle, Ellipse, Polygon

    tipo = info["tipo"]
    obj = info["objeto"]
    color = info["color"]
    label = info["label"]
    scale = info.get("scale", 1.0)

    if tipo == "rect":
        x = obj["left"] / scale
        y = obj["top"] / scale
        w = obj["width"] / scale
        h = obj["height"] / scale
        return Rectangle((x, y), w, h, edgecolor=color, facecolor='none', linewidth=1.5, label=label)

    elif tipo == "circle":
        x = obj["left"] / scale
        y = obj["top"] / scale
        w = obj["width"] / scale
        h = obj["height"] / scale
        cx = x + w / 2
        cy = y + h / 2
        return Ellipse((cx, cy), w, h, edgecolor=color, facecolor='none', linewidth=1.5, label=label)

    elif tipo == "path":
        points = [(p[1] / scale, p[2] / scale) for p in obj["path"]]
        return Polygon(points, closed=True, edgecolor=color, facecolor='none', linewidth=1.5, label=label)

    return None

#EDICIONIMAGENES
def inicializar_sesion_rgb():
    for k in ("F_r", "F_g", "F_b", "Ffilt_r", "Ffilt_g", "Ffilt_b"):
        if k not in st.session_state:
            st.session_state[k] = None

def filtro_canal(canal, canal_name, filtros_disponibles, key_prefix):
    st.subheader(f"Canal {canal_name}")
    tipo_filtro = st.selectbox(f"Filtro {canal_name}", filtros_disponibles, key=f"freqTipo{key_prefix}")

    fc1 = st.slider(f"Frecuencia relativa de corte 1 ({canal_name})", 0.0, 1.0, 0.01, key=f"fc1{key_prefix}")
    fc2 = None
    orden = None

    if tipo_filtro == "Pasa banda":
        fc2 = st.slider(f"Frecuencia relativa de corte 2 ({canal_name})", fc1 + 0.01, 1.0, fc1 + 0.01, key=f"fc2{key_prefix}")
    if tipo_filtro == "Butterworth":
        orden = st.slider("Orden", 1, 10, 1, key=f"orden{key_prefix}")

    # Vista previa automática
    _, _, previa_Ffilt, previa_Ifilt = filtro_frecuencia(canal, tipo_filtro, fc1, fc2, orden)
    fig_tf, ax_tf = plt.subplots(figsize=(7, 5))
    ax_tf.imshow(previa_Ffilt, cmap="gray")
    ax_tf.axis("off")
    st.write(f"TF Filtrada {canal_name} (Previsualización)")
    st.pyplot(fig_tf)

    fig_img, ax_img = plt.subplots(figsize=(7, 5))
    ax_img.imshow(previa_Ifilt, cmap="gray")
    ax_img.axis("off")
    st.write(f"Img Filtrada {canal_name} (Previsualización)")
    st.pyplot(fig_img)

    if st.button(f"Aplicar Frecuencia {key_prefix}", key=f"AF_{key_prefix}"):
        F, mask, Ffilt, img_filtrada = filtro_frecuencia(canal, tipo_filtro, fc1, fc2, orden)
        st.session_state[f"F_{key_prefix}"] = F
        st.session_state[f"Ffilt_{key_prefix}"] = Ffilt
        st.session_state[f"canal{canal_name}"] = fn.normalizar_0_255(img_filtrada)

def mostrar_canales_rgb():
    columnas = st.columns(3)
    canales = [("R", "canalR"), ("G", "canalG"), ("B", "canalB")]
    for col, (name, key) in zip(columnas, canales):
        with col:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(st.session_state[key], cmap='gray')
            ax.axis('off')
            st.write(f"Canal {name}")
            st.pyplot(fig)

def recombinar_canales(img_t):
    if all(k in st.session_state for k in ("canalR", "canalG", "canalB")):
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
            col1, col2 = st.columns(2)
            with col1:
                fig_orig, ax_orig = plt.subplots(figsize=(15, 5))
                ax_orig.imshow(img_t, cmap='gray')
                ax_orig.axis('off')
                st.write("Imagen Original")
                st.pyplot(fig_orig)

            with col2:
                fig_recom, ax_recom = plt.subplots(figsize=(15, 5))
                ax_recom.imshow(imagen_recombinada)
                ax_recom.axis('off')
                st.write("Imagen Final")
                st.pyplot(fig_recom)

            st.image(imagen_recombinada, caption="Imagen recombinada", clamp=True, use_container_width=True)

        st.session_state["img_analisis"] = imagen_recombinada

def rotar_gris():
    st.header("🎛️ Rotación")
    c1, c2, c3 = st.columns(3)
    with c2:
        st.subheader("Canal Gris")
        angulo_gray = st.slider("Ángulo de rotación", min_value=-180, max_value=180, value=0, step=1)
        previa_Ifilt_gray = scipy.ndimage.rotate(st.session_state["canalGray"], angle=angulo_gray, reshape=False, mode='nearest')

        figPrevgray, axPrevgray = plt.subplots(figsize=(7, 5))
        axPrevgray.imshow(previa_Ifilt_gray, cmap='gray')
        axPrevgray.axis("off")
        st.write("Img Gris (Previsualización)")
        st.pyplot(figPrevgray)

        if st.button("Rotar"):
            st.session_state["canalGray"] = scipy.ndimage.rotate(st.session_state["canalGray"], angle=angulo_gray, reshape=False, mode='nearest')

def main(img_t, img_a):
    if img_a.ndim == 3:  # RGB
        inicializar_sesion_rgb()
        st.header("🌌 Filtros en el Dominio Frecuencial por Canal")
        filtros_r = ["Pasa bajo", "Pasa alto", "Pasa banda", "Gaussiano", "Pasa alto Gaussiano", "Rampa", "Parzen", "Shepp-Logan", "Hann", "Hamming", "Butterworth"]
        filtros_gb = ["Pasa bajo", "Pasa alto", "Pasa banda", "Gaussiano"]

        f1, f2, f3 = st.columns(3)
        with f1:
            filtro_canal(st.session_state["canalR"], "Rojo", filtros_r, "r")
        with f2:
            filtro_canal(st.session_state["canalG"], "Verde", filtros_gb, "g")
        with f3:
            filtro_canal(st.session_state["canalB"], "Azul", filtros_gb, "b")

        mostrar_canales_rgb()
        recombinar_canales(img_t)

def mostrar_figura_como_imagen(fig, ancho=None):
    """
    Convierte una figura de Matplotlib en una imagen PNG y la muestra con st.image.

    Parámetros:
    - fig: figura de matplotlib (creada con plt.subplots())
    - ancho: ancho opcional en píxeles para mostrar la imagen en Streamlit
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    st.image(img, width=ancho)



 # Función para calcular puntos extremos de una línea dada su dirección


def puntos_extremos(cx, cy, ang_deg, L):
    theta = np.radians(ang_deg)
    dx = L * np.cos(theta)
    dy = L * np.sin(theta)
    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy
    return (x1, y1, x2, y2)

import hashlib

def hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def hash_imagen(img_np):
    if img_np is None:
        return None
    # Convertimos siempre a uint8 y aplanamos para que el hash sea estable
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    # Flatten y hasheo
    return hashlib.md5(img_np.tobytes()).hexdigest()

def plot_histograma_estilo_matlab(canal, color='k', etiqueta=None):
    canal = canal[np.isfinite(canal)]
    bins = np.histogram_bin_edges(canal, bins='auto')
    hist, bin_edges = np.histogram(canal, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

def calcular_fwhm(x, y):
    if y is None or len(y) == 0:
        return None, None, None, None

    y = np.array(y)
    x = np.array(x)

    max_idx = np.argmax(y)
    max_y = y[max_idx]
    half_max = max_y / 2

    # Búsqueda hacia la izquierda desde el pico
    izquierda = None
    for i in range(max_idx - 1, -1, -1):
        if y[i] < half_max <= y[i + 1] or y[i] > half_max >= y[i + 1]:
            frac = (half_max - y[i]) / (y[i + 1] - y[i])
            izquierda = x[i] + frac * (x[i + 1] - x[i])
            break

    # Búsqueda hacia la derecha desde el pico
    derecha = None
    for i in range(max_idx, len(y) - 1):
        if y[i] >= half_max > y[i + 1] or y[i] <= half_max < y[i + 1]:
            frac = (half_max - y[i]) / (y[i + 1] - y[i])
            derecha = x[i] + frac * (x[i + 1] - x[i])
            break

    if izquierda is not None and derecha is not None:
        fwhm = abs(derecha - izquierda)
        return fwhm, izquierda, derecha, half_max
    else:
        return None, None, None, None
    
def RLE2(A, B, C=None, plot=True, titulo="B vs A", xlabel="A, []", ylabel="B, []"):

    n = len(A)
    a = (n * np.sum(A * B) - np.sum(A) * np.sum(B)) / (n * np.sum(A**2) - np.sum(A)**2)
    b = (np.sum(B) - a * np.sum(A)) / n
    R = abs(np.sum((A - np.mean(A)) * (B - np.mean(B))) /
            (np.sqrt(np.sum((A - np.mean(A))**2)) * np.sqrt(np.sum((B - np.mean(B))**2))))
    
    Y_est = a * A + b
    residuos = B - Y_est
    SR = np.sum(residuos**2)
    VR = SR / (n - 2)
    D = np.sqrt(VR)
    Sa = D * np.sqrt(n / (n * np.sum(A**2) - np.sum(A)**2))
    Sb = D * np.sqrt(np.sum(A**2) / (n * np.sum(A**2) - np.sum(A)**2))

    fig = None
    if plot:
        X = np.linspace(min(A) - 0.1, max(A) + 0.1, 200)
        Y = a * X + b
        fig, ax = plt.subplots()
        ax.plot(X, Y, 'r-', label="Recta estimada")
        ax.plot(A, B, 'bo', label="Datos")
        
        if C is not None:
            ax.errorbar(A, B, yerr=C, fmt='none', ecolor='black', capsize=4, linestyle='None', label="Error")

        ax.set_xlabel(xlabel if xlabel is not None else "A")
        ax.set_ylabel(ylabel if ylabel is not None else "B")
        ax.set_title(titulo if titulo is not None else "Regresión lineal B vs A")
        ax.legend()
        ax.grid(False)

    resultados = {
        'a': a, 'b': b, 'R': R, 'D': D, '𝛥a': Sa, '𝛥b': Sb,
        'fig': fig
    }
    return resultados

def RLE_polyfit(U, V, grado, W=None, plot=True, titulo="", xlabel="", ylabel="", use_weights=True, show_errorbars=False):
    U = np.array(U)
    U_original = U.copy()
    #K=st.checkbox("Escalar U, [-1,1]",key="K1")
    #if K:
    x_min, x_max = np.min(U), np.max(U)
    U = 2 * (U - x_min) / (x_max - x_min) - 1
    #escalado = True
    #else:
    #    escalado = False

    #x_mean = np.mean(U)
    #x_range = np.max(U) - np.min(U)
    #U = (U - x_mean) / x_range
    V = np.array(V)
    n = len(U)

    # Manejo pesos según use_weights y W válido
    if use_weights and W is not None and len(W) == n:
        pesos = 1 / np.array(W)**2
    else:
        pesos = None

    # Ajuste polinomial (ponderado o no)
    coef = np.polyfit(U, V, grado, w=pesos)
    V_fit = np.polyval(coef, U)
    residuos = V - V_fit

    # R²
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    # Matriz diseño para incertidumbre
    A = np.vander(U, grado + 1, increasing=False)

    # Covarianza coeficientes
    if pesos is not None:
        W_diag = np.diag(pesos)
        cov = np.linalg.inv(A.T @ W_diag @ A)
        sigma2 = ss_res / (n - grado - 1)
        cov = cov * sigma2
    else:
        cov = np.linalg.inv(A.T @ A)
        sigma2 = ss_res / (n - grado - 1)
        cov = cov * sigma2
    
    delta_coef = np.sqrt(np.diag(cov))

    # Error estándar residual
    D = np.sqrt(ss_res / (n - grado - 1))

    fig = None
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(U_original, V, color="r", label="Datos")
        #if K:
        U_plot_orig = np.linspace(x_min, x_max, 1000)
        U_plot = 2 * (U_plot_orig - x_min) / (x_max - x_min) - 1
        #else:
        #    U_plot_orig = np.linspace(np.min(U), np.max(U), 1000)
        #    U_plot = U_plot_orig
        V_plot = np.polyval(coef, U_plot)
        from scipy.interpolate import lagrange
        #U_plot = lagrange(U, V)
        #V_plot = U_plot(U)
        ax.plot(U_plot_orig, V_plot, color="b", label=f"Polinomio grado {grado}")

        # Mostrar barras de error si está pedido y W válido
        if show_errorbars and W is not None and len(W) == n:
            ax.errorbar(U_original, V, yerr=W, fmt='none', ecolor='black', capsize=3, label="Errores")

        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.legend()
        ax.grid(False)
        #min_value = float(min(V_plot) - (np.max(W) if W is not None else 10.0))
        #max_value = float(max(V_plot) + (np.max(W) if W is not None else 10.0))
        ylimit2 = st.number_input(
                            "Extremo inferior en (y) visible",
                            value=float(min(V_plot)),      # valor por defecto
                            min_value=float(min(V_plot)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                            max_value=float(max(V_plot)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                            step=0.01,
                            format="%.4f",key="yl21")
        ylimit1 = st.number_input(
                            "Extremo superior en (y) visible",
                            value=float(max(V_plot)),      # valor por defecto
                            min_value=float(min(V_plot)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                            max_value=float(max(V_plot)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                            step=0.01,
                            format="%.4f",key="yl11")
        plt.ylim([ylimit2,ylimit1])
        plt.close(fig)

    resultado = {
        'coef': coef,
        'delta_coef': delta_coef,
        'residuos': residuos,
        'R2': R2,
        'error': np.linalg.norm(residuos),
        'D': D,
        'fig': fig,
    }

    if grado == 1:
        resultado['a'] = coef[0]
        resultado['b'] = coef[1]
        resultado['𝛥a'] = delta_coef[0]
        resultado['𝛥b'] = delta_coef[1]
        resultado['R'] = np.sqrt(R2)

    return resultado

def RLE_polyfit2(volu1,volu2,U, V, grado, W=None, plot=True, titulo="", xlabel="", ylabel="", use_weights=True, show_errorbars=False):
    U = np.array(U)
    U = np.array(U)
    U_original = U.copy()
    #K=st.checkbox("Escalar U, [-1,1]",key="K1")
    #if K:
    x_min, x_max = np.min(U), np.max(U)
    U = 2 * (U - x_min) / (x_max - x_min) - 1
    #escalado = True
    #else:
    #    escalado = False
    V = np.array(V)
    n = len(U)

    # Manejo pesos según use_weights y W válido
    if use_weights and W is not None and len(W) == n:
        pesos = 1 / np.array(W)**2
    else:
        pesos = None

    # Ajuste polinomial (ponderado o no)
    coef = np.polyfit(U, V, grado, w=pesos)
    V_fit = np.polyval(coef, U)
    residuos = V - V_fit

    # R²
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    # Matriz diseño para incertidumbre
    A = np.vander(U, grado + 1, increasing=False)

    # Covarianza coeficientes
    if pesos is not None:
        W_diag = np.diag(pesos)
        cov = np.linalg.inv(A.T @ W_diag @ A)
        sigma2 = ss_res / (n - grado - 1)
        cov = cov * sigma2
    else:
        cov = np.linalg.inv(A.T @ A)
        sigma2 = ss_res / (n - grado - 1)
        cov = cov * sigma2
    
    delta_coef = np.sqrt(np.diag(cov))

    # Error estándar residual
    D = np.sqrt(ss_res / (n - grado - 1))

    fig = None
    if plot:
        fig, ax = plt.subplots()
        #ax.scatter(U_original, V, color="r", label="Datos")
        #if K:
        U_plot_orig = np.linspace(x_min, x_max, 1000)
        U_plot = 2 * (U_plot_orig - x_min) / (x_max - x_min) - 1
        #else:
        #    U_plot_orig = np.linspace(np.min(U), np.max(U), 1000)
        #    U_plot = U_plot_orig
        V_plot = np.polyval(coef, U_plot)
        from scipy.interpolate import lagrange
        #U_plot = lagrange(U, V)
        #V_plot = U_plot(U)
        ax.plot(U_plot_orig, V_plot, color="b", label=f"Polinomio grado {grado}")

        # Mostrar barras de error si está pedido y W válido
        #if show_errorbars and W is not None and len(W) == n:
        #    ax.errorbar(U, V, yerr=W, fmt='none', ecolor='black', capsize=3, label="Errores")

        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.legend()
        ax.grid(False)

        with volu1:
            ylimit2 = st.number_input(
                                "Extremo inferior en (y) visible",
                                value=float(min(V_plot)),      # valor por defecto
                                min_value=float(min(V_plot)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                                max_value=float(max(V_plot)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                                step=0.01,
                                format="%.4f",key="yl221")
            ylimit1 = st.number_input(
                                "Extremo superior en (y) visible",
                                value=float(max(V_plot)),      # valor por defecto
                                min_value=float(min(V_plot)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                                max_value=float(max(V_plot)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                                step=0.01,
                                format="%.4f",key="yl211")
        plt.ylim([ylimit2,ylimit1])
        
        plt.close(fig)

    resultado = {
        'coef': coef,
        'delta_coef': delta_coef,
        'residuos': residuos,
        'R2': R2,
        'error': np.linalg.norm(residuos),
        'D': D,
        'fig': fig,
        "U_plot": U_plot_orig,
        "V_plot": V_plot,
    }

    if grado == 1:
        resultado['a'] = coef[0]
        resultado['b'] = coef[1]
        resultado['𝛥a'] = delta_coef[0]
        resultado['𝛥b'] = delta_coef[1]
        resultado['R'] = np.sqrt(R2)

    return resultado

def formatear_ecuacion(coefs):
    """
    Recibe lista de coeficientes [a_n, a_{n-1}, ..., a_0]
    Devuelve string LaTeX con la ecuación polinómica,
    por ejemplo: y = a_n x^n + a_{n-1} x^{n-1} + ... + a_0
    """
    grado = len(coefs) - 1
    terminos = []

    for i, a in enumerate(coefs):
        exp = grado - i
        signo = '+' if a >= 0 else '-'
        valor = abs(a)

        # Formatear coeficiente con 5 decimales
        coef_str = f"{valor:.5f}"

        if exp == 0:
            termino = f"{coef_str}"
        elif exp == 1:
            termino = f"{coef_str} x"
        else:
            termino = f"{coef_str} x^{exp}"

        # El primer término no debe llevar signo + al inicio
        if i == 0:
            # Si el coef es negativo, anteponer -
            if signo == '-':
                terminos.append(f"- {termino}")
            else:
                terminos.append(termino)
        else:
            terminos.append(f" {signo} {termino}")

    # Armar la ecuacion completa
    ecuacion = "y = " + "".join(terminos)
    # Escapar para LaTeX en st.latex (colocar dentro de $...$ no hace falta en st.latex)
    return ecuacion

def formato_coeficiente(coef, delta, i, grado):
    # Forzar signo explícito para coef
    signo_coef = '+' if coef >= 0 else '-'
    valor_coef = abs(coef)
    
    # Formatear coef y delta con 5 decimales
    coef_str = f"{signo_coef}{valor_coef:.5f}"
    delta_str = f"{delta:.5f}"
    
    # Regreso la línea con espacios LaTeX claros y el signo menos correcto
    # Nota: el signo menos es '-' normal para LaTeX en modo math
    return rf"a_{{{grado - i}}} = {coef_str} \quad \Delta a_{{{grado - i}}} = {delta_str}"


def integral(f, a, b, h, tipo="trapecio"):
    x = np.arange(a, b + h, h)  # asegura incluir b
    if tipo == "trapecio2":
        y = f(x)
        return (y[0] + 2 * np.sum(y[1:-1]) + y[-1]) * h / 2
    elif tipo=="trapecio":
        rango=np.arange(a+h,b,h)
        F=0
        for k in rango:
            F=f(k)+F
        F=(f(a)+2*F+f(b))*h/2
        return F
    elif tipo == "simpson2menosrobusto":
        if len(x) % 2 == 0:  # Simpson requiere cantidad impar de puntos
            x = x[:-1]
        y = f(x)
        return (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1]) * h / 3
    elif tipo == "riemann":
        x_mid = np.arange(a + h / 2, b, h)
        return np.sum(f(x_mid)) * h
    elif tipo == "simpson2":
        n = int((b - a) / h)
        if n % 2 != 0:
            n += 1  # Aseguramos que haya número par de subintervalos
        x = np.linspace(a, b, n+1) # Usamos linspace porque arange es menos preciso
        y = f(x) # Dedinimos la función
        R = (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))*h/3
        return R
    elif tipo=="simpson":
        n = int((b - a) / h)
        if n % 2 != 0:
            n += 1  # Aseguramos que haya número par de subintervalos
        h = (b-a)/n # corrijo h
        F = 0  # suma en los puntos pares (excepto extremos)
        H = 0  # suma en los impares
        for k in range(1, n, 2):  # impares
            xk = a + k*h
            H += f(xk)
        for l in range(2, n-1, 2):  # pares (excluye extremos)
            xl = a + l*h
            F += f(xl)
        R = (f(a) + 2*F + 4*H + f(b)) * h / 3
        return R

# Función común para carga de datos con columnas U, V, (opcionalmente W)
def cargar_datos_uv_w():
    W = None
    archivo = st.file_uploader("Subí un archivo CSV o Excel con columnas U, V (y opcionalmente W)", type=["csv", "xlsx"])

    df = None
    U = V = W = None

    if archivo is not None:
        try:
            if archivo.name.endswith(".csv"):
                df = pd.read_csv(archivo)
            elif archivo.name.endswith(".xlsx"):
                df = pd.read_excel(archivo)
            else:
                st.error("Formato de archivo no soportado.")
                return None, None, None

            if 'U' not in df.columns or 'V' not in df.columns:
                st.error("El archivo debe tener columnas 'U' y 'V'.")
                return None, None, None

            U = df['U'].values
            V = df['V'].values
            W = df['W'].values if 'W' in df.columns else None
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None, None, None
    else:
        st.subheader("Carga manual de datos")
        U_txt = st.text_area("Valores de U (separados por coma)", "5, 7, 3, 9, 11, 14")
        V_txt = st.text_area("Valores de V (separados por coma)", "28.1, 41.8, 14, 56, 70, 91.2")
        W_txt = st.text_area("Errores (W) opcionales para V", "2.2, 2, 1.5, 2, 1.4, 1")

        try:
            U = np.fromstring(U_txt, sep=",")
            V = np.fromstring(V_txt, sep=",")
            W = np.fromstring(W_txt, sep=",") if W_txt.strip() != "" else None
        except:
            st.error("Error al convertir los datos. Verificá el formato.")
            return None, None, None

    return U, V, W


from scipy.optimize import curve_fit

def RLE_exponential_fit(U, V, W=None, plot=True, titulo="", xlabel="", ylabel="", use_weights=True, show_errorbars=False):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    n = len(U)
    if len(U) == 0 or len(V) == 0:
        st.error("Error: las listas de datos U y V no pueden estar vacías.")
        return {}
    # Modelo: y = a * exp(b * x) + c
    def modelo_exp(x, a, b, c):
        return a * np.exp(b * x) + c

    # Validación de pesos
    if use_weights and W is not None and len(W) == n:
        sigma = np.array(W, dtype=float)
        if np.any(sigma <= 0):
            st.warning("Los errores (W) deben ser positivos. Se ignorarán.")
            sigma = None
    else:
        sigma = None

    # Verificaciones básicas
    if len(U) != len(V):
        st.error(f"U y V deben tener la misma longitud: {len(U)} ≠ {len(V)}")
        return {}

    # Estimación inicial inteligente
    a0 = max(V) - min(V)
    c0 = min(V)
    pendiente_aprox = (V[-1] - V[0]) / (U[-1] - U[0])
    b0 = 1.0 if pendiente_aprox > 0 else -1.0
    p0 = [a0, b0, c0]

    # Ajuste con curve_fit
    try:
        popt, pcov = curve_fit(
            modelo_exp, U, V,
            p0=p0,
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=10000
        )
    except RuntimeError as e:
        st.error(f"No se pudo ajustar el modelo exponencial: {e}")
        return {}

    a, b, c = popt
    delta_a, delta_b, delta_c = np.sqrt(np.diag(pcov))

    # Residuos y métricas
    V_fit = modelo_exp(U, a, b, c)
    residuos = V - V_fit
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((V - np.mean(V))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    D = np.sqrt(ss_res / (n - 3))  # 3 parámetros ajustados

    ylimit2 = st.number_input(
                        "Extremo inferior en (y) visible",
                        value=float(min(V)),      # valor por defecto
                        min_value=float(min(V)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                        max_value=float(max(V)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                        step=0.01,
                        format="%.4f",key="yl21")
    ylimit1 = st.number_input(
                        "Extremo superior en (y) visible",
                        value=float(max(V)),      # valor por defecto
                        min_value=float(min(V)-(np.max(W)+10.0 if W is not None else 10.0)),  # valor mínimo que se puede elegir
                        max_value=float(max(V)+(np.max(W)+10.0 if W is not None else 10.0)),  # valor máximo que se puede elegir
                        step=0.01,
                        format="%.4f",key="yl11")
    
    # Gráfico
    fig = None
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(U, V, color="r", label="Datos")
        U_plot = np.linspace(min(U), max(U), 300)
        V_plot = modelo_exp(U_plot, a, b, c)
        ax.plot(U_plot, V_plot, color="blue", label="Ajuste exponencial")

        if show_errorbars and sigma is not None:
            ax.errorbar(U, V, yerr=sigma, fmt='none', ecolor='black', capsize=3)

        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.legend()
        #ax.grid(True)
        plt.ylim([ylimit2,ylimit1])
        plt.close(fig)
    fig2 = None
    if plot:
        fig2, ax = plt.subplots()
        #ax.scatter(U, V, color="r", label="Datos")
        U_plot = np.linspace(min(U), max(U), 300)
        V_plot = modelo_exp(U_plot, a, b, c)
        ax.plot(U_plot, V_plot, color="blue", label="Ajuste exponencial")

        #if show_errorbars and sigma is not None:
        #    ax.errorbar(U, V, yerr=sigma, fmt='none', ecolor='black', capsize=3)

        ax.set_title(titulo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.legend()
        #ax.grid(True)
        plt.close(fig2)
    f_evaluable = lambda x: a * np.exp(b * x) + c
    U_plot = np.linspace(min(U), max(U), 300)
    V_plot = modelo_exp(U_plot, a, b, c)
    df = lambda x: a * b * np.exp(b * x)
    # Resultado
    return {
        'a': a,
        'b': b,
        'c': c,
        '𝛥a': delta_a,
        '𝛥b': delta_b,
        '𝛥c': delta_c,
        'residuos': residuos,
        'R2': R2,
        'R': np.sqrt(R2),
        'D': D,
        'fig': fig,
        'fig2': fig2,
        'f': f_evaluable,
        'x_plot': U_plot,
        'y_plot': V_plot,
        'df': df,
        'modelo': 'exponencial'
    }

def biner2(g,dg,a,b,n,e):
  """biner(x,f,a,b,n):
  [x: símbolo]; [f: función simbólica];
  [a: extremo inferior]; [b: extremo superior];
  [n: número de iteraciones]; [e: máximo error esperado]"""
  #df = sp.diff(f, x)
  #g = sp.lambdify(x, f, "math")   # función evaluable
  #dg = sp.lambdify(x, df, "math") # derivada evaluable
  # Comprobación de signo contrario
  if g(a)*g(b) >= 0:
      raise ValueError("Rango incompatible: no hay cambio de signo en el intervalo.")
  # Primera parte: Bisección
  for k in range(n+1):
      c = (a + b)/2
      if g(a)*g(c) < 0:
          b = c
      else:
          a = c
  # Segunda parte: Newton-Raphson
  m = 999999
  l = 0
  while m >= e:
      z = c
      c = c - g(c)/dg(c)
      m = abs(c - z)
      l += 1
  return c, np.array([a, b]), l

