Proyecto de Visión Computacional
======================================================

**Introducción**
----------------

Este proyecto se desarrolló durante la **Semana Tec** con el objetivo de aprender y aplicar conceptos clave de **visión computacional** utilizando Python y OpenCV. El trabajo se enfocó en dos áreas principales: **convolución de imágenes** y **reconocimiento de placas**, integrando herramientas de control de versiones a través de GitHub para fomentar la colaboración en equipo y las buenas prácticas de desarrollo.

Cada estudiante trabajó en su propio repositorio y luego unió sus ramas en un repositorio de equipo.

**Requerimientos del Proyecto**
-------------------------------

### **Software**

-   **Python 3.x**
-   **Librerías necesarias**:
    -   OpenCV: `pip install opencv-python-headless`
    -   EasyOCR: `pip install easyocr`
    -   Matplotlib: `pip install matplotlib`

### **Archivos Necesarios**

-   Imágenes en formato `.jpg` o `.png` en la carpeta `images/` para el reconocimiento de placas.

**Implementación de Convolución**
==========================================================

Estos archivos forman parte de la implementación del concepto de **convolución** aplicado al procesamiento de imágenes. A continuación, se presenta una breve descripción de los archivos principales y su funcionalidad:

* * * * *

### **1\. Archivo: `convolution.py`**

Este archivo contiene las funciones principales para aplicar **convoluciones** en imágenes usando kernels personalizables. Las convoluciones son una operación clave para modificar imágenes mediante filtros, mejorando bordes, reduciendo ruido o detectando líneas.

#### **Funciones destacadas:**

-   **`conv_helper(fragment, kernel)`**:\
    Realiza la multiplicación entre fragmentos de la imagen y el kernel para devolver la suma resultante.

-   **`convolution(image, kernel, padding=True)`**:\
    Aplica la convolución con o sin **padding** para gestionar bordes de la imagen, devolviendo la imagen filtrada.

-   **`normalize_image(image)`**:\
    Ajusta los valores de la imagen para normalizarlos entre 0 y 255.

-   **`enhance_contrast(image)`**:\
    Mejora el contraste de la imagen utilizando **ecualización de histograma**.

-   **`apply_filter(image_path, filter_type, contrast_enhance=True)`**:\
    Lee una imagen desde la ruta proporcionada, aplica el filtro seleccionado y muestra los resultados en pantalla.

#### **Filtros incluidos:**

-   **Sobel Horizontal y Vertical**: Resalta bordes en diferentes direcciones.
-   **Laplaciano**: Detecta bordes en todas las direcciones.
-   **Gaussian Blur**: Aplica un desenfoque gaussiano para reducir ruido.
-   **Edge Detection**: Realza bordes con un filtro laplaciano ajustado.
-   **Line Detection (Horizonte y Diagonales)**: Detecta líneas específicas en varias direcciones.

* * * * *

### **2\. Archivo: `sobel_test.py`**

Este archivo está dedicado a probar el **filtro Sobel**, que se usa para detectar bordes horizontales y verticales en imágenes. El Sobel es ideal para operaciones de preprocesamiento en sistemas de visión que buscan detectar contornos o características estructurales.

#### **Uso:**

`python convolution.py --image images/house.jpg --filter sobel_horizontal`

Este comando aplica un filtro **Sobel Horizontal** a la imagen especificada y muestra los resultados.

* * * * *

### **3\. Archivo: `gaussian_test.py`**

Este archivo implementa una versión del **Gaussian Blur** utilizando un kernel más grande y poderoso. Esta técnica es útil para reducir el ruido en imágenes antes de aplicar otros filtros.

#### **Uso:**

`apply_filter('images/house.jpg', 'gaussian_blur')`

Este ejemplo aplica un desenfoque gaussiano a la imagen seleccionada y muestra el resultado.

* * * * *

### **4\. Archivo: `edge_detection.py`**

Este archivo ejecuta un filtro especializado en **detección de bordes**. Utiliza un filtro Laplaciano modificado para detectar cambios abruptos en los niveles de intensidad de la imagen.

#### **Uso:**


`python convolution.py --image images/placa.png --filter edge_detection`

Este comando aplica detección de bordes a una imagen de placa vehicular, útil para preprocesar imágenes antes de OCR.

* * * * *

### **5\. Pruebas y Resultados**

Los ejemplos de este proyecto permiten explorar cómo diferentes filtros afectan las imágenes de entrada, proporcionando una experiencia práctica en el uso de **kernels** en visión computacional. Los resultados se muestran en ventanas gráficas utilizando **Matplotlib**.

**Reconocimiento de Placas**
=====================================================

Este conjunto de archivos implementa un sistema de **reconocimiento de placas vehiculares** utilizando **OpenCV** y **EasyOCR**. A continuación, se presenta una descripción detallada de los archivos y sus funcionalidades.

* * * * *

### **1\. Archivo: `plate_recognition.py`**

Este es el archivo principal que gestiona el flujo completo del reconocimiento de placas, desde la lectura de la imagen hasta la extracción del texto utilizando OCR (Reconocimiento Óptico de Caracteres).

#### **Funciones destacadas:**

-   **`detectar_contorno_placa`**: Detecta y selecciona el contorno que corresponde a una placa de matrícula en la imagen. Se basa en la forma de un cuadrilátero.

-   **`extraer_region_placa`**: Recorta la región de la placa de la imagen original utilizando el contorno detectado.

-   **`preprocesar_imagen`**: Mejora la precisión del OCR mediante técnicas de preprocesamiento, incluyendo conversión a escala de grises, aumento de contraste y binarización.

-   **`obtener_texto_placa`**: Utiliza **EasyOCR** para detectar y extraer el texto de la placa preprocesada. El texto se filtra y ajusta para corregir errores comunes de reconocimiento.

-   **`sobreponer_texto`**: Muestra el texto detectado directamente en la imagen, resaltando también el contorno de la placa.

-   **`mostrar_imagenes`**: Visualiza diferentes etapas del proceso, como la imagen original, la imagen preprocesada y el resultado final con el texto sobrepuesto.

* * * * *

### **Uso de EasyOCR y OpenCV**

El proceso comienza aplicando técnicas de visión computacional con **OpenCV** para:

-   **Filtrar ruido**: Usando un filtro bilateral.
-   **Detectar bordes**: Con el detector de bordes Canny.
-   **Extraer el contorno**: A través de la aproximación de contornos en la imagen.

Una vez aislada la región de la placa, **EasyOCR** detecta el texto en la imagen procesada.

* * * * *

### **Funciones de Convolución y Filtros Adicionales**

El proyecto también incluye varios **filtros de convolución** que ayudan a mejorar la detección de características:

-   **Sobel Horizontal y Vertical**: Para resaltar bordes específicos.
-   **Laplaciano**: Para detectar bordes más complejos.
-   **Gaussian Blur**: Reduce el ruido en la imagen.

Estos filtros se aplican durante el preprocesamiento para mejorar la calidad del reconocimiento de caracteres.

* * * * *

### **Requisitos de Instalación**

-   Python 3.x
-   OpenCV
-   EasyOCR
-   Matplotlib

#### **Instalación de Dependencias:**

`pip install opencv-python-headless easyocr matplotlib`

* * * * *

### **Ejecución del Proyecto**

1.  **Preparación del entorno**: Asegúrate de tener las dependencias instaladas.

2.  **Coloca la imagen** que contiene la placa en la carpeta `images/`.

3.  **Ejecuta el script principal**:

    `python plate_recognition.py`

4.  **Verifica los resultados**:\
    Se mostrarán las diferentes etapas del procesamiento, incluyendo la imagen original, la imagen con la placa detectada y la imagen final con el texto extraído.