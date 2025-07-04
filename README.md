# DaTrix
**Datrix** es una app web desarrollada con **Python** y **Streamlit**, orientada estudiantes de **Ingeniería**, **Física Médica**, **Diagnóstico por Imágenes**.

Consta de 3 módulos principales:

---

## Álgebra
Permite realizar:
- Operaciones con matrices
- Potencia, producto, suma por un escalar.
- Rango, transpuesta, determinante, inversa, autovalores y autovectores.
- Suma, resta, producto matricial. Producto, cociente bin a bin. Conmutador.
- Resolución de sistemas lineales, ajuste polnómico, ajuste exponencial, interpolación spline. Análisis de curvas, máximo y mínimo absoluto, integración, resolución de ecuaciones no lineales.

## Procesamiento de Imágenes
Incluye herramientas para:
- Visualización y ventaneo (como en imágenes de tomografía).
- Edición: rotación, brillo, contraste, filtrado espacial y frecuencial, operaciones morfológicas, ecualización.
- Análsis de perfil de intensidad.
- Análisis de regiones de interés (ROI).
- Curvas ROI vs. tiempo para archivos multiframe (por ejemplo, en estudios dinámicos).

## Reconstrucción Tomográfica
Simulador de reconstrución tomográfica de fantomas matemáticos.

### Métodos simulados
- Filtered Back Projection (**FBP**)
- Maximum Likelihood Expectation Maximization (**MLEM**)
- Ordered Subset Expectation Maximization (**OSEM**)
- Simultaneous Algebraic Reconstruction Technique (**SART**).

### Modos disponibles
- Computed Tomography (**CT**)
- Single Photon Emission Computed Tomography (**SPECT**).

### Parámetros configurables
- Ángulo inicial
- Ángulo final
- Paso angular
- Número de iteraciones
- Número de subsets.

---
## Capturas de pantalla

### Álgebra

**Operaciones con matrices**

<img src="capturas/opmatrices.jpg" width="600"/>

**Ajustes y análisis de curva**

<img src="capturas/sistemas.jpg" width="600"/>
<img src="capturas/analisdecurva.jpg" width="600"/>

### Procesamiento de Imágenes

**Visualizador**

<img src="capturas/visualizador.jpg" width="600"/>

**Editor**

<img src="capturas/editor.jpg" width="600"/>

**Calibrador**

<img src="capturas/calibrador.jpg" width="600"/>

**Análisis de Perfil**

<img src="capturas/perfilejemplo.jpg" width="600"/>

**Curva ROI vs. tiempo**

<img src="capturas/ROISejemplo.jpg" width="600"/>

### Reconstrucción Tomográfica

**Ejemplo: modo SPECT, método FBP**

<img src="capturas/modulo3.jpg" width="600"/>

---

## Enlace a la app

👉 [Ir a app]: https://datrix.streamlit.app/

---

## Futuras mejoras

- **Álgebra:** transformada de Fourier para curvas.
- **Procesamiento de Imágenes:** transformada de Fourier para perfil de intensidad y curva ROI vs. tiempo.
- **Reconstrucción tomográfica:** simulación de ruido, gráfico de error cuadrático medio normalizado.
 
---

## Autor

**David Fernández Basti**

*Técnico Universitario en Diagnóstico por Imágenes*

*Estudiante de Ingeniería Nuclear.*
