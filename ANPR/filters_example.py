import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr

def plotImages(img1, img2, title1='Imagen Original', title2='Imagen Modificada', cmap2='gray'):
    """
    Muestra dos imágenes lado a lado con sus respectivos títulos.

    Args:
        img1 (np.ndarray): Imagen original en color.
        img2 (np.ndarray): Imagen modificada (puede ser en escala de grises o binarizada).
        title1 (str, optional): Título para la primera imagen. Predeterminado es 'Imagen Original'.
        title2 (str, optional): Título para la segunda imagen. Predeterminado es 'Imagen Modificada'.
        cmap2 (str, optional): Mapa de colores para la segunda imagen. Predeterminado es 'gray'.
    """
    plt.figure(figsize=(10, 5))

    # Imagen Original
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis('off')

    # Imagen Modificada
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2)
    plt.title(title2)
    plt.axis('off')

    plt.show()

def conv_helper(fragment, kernel):
    """
    Multiplica dos matrices y devuelve su suma.

    Args:
        fragment (np.ndarray): Fragmento de la imagen.
        kernel (np.ndarray): Núcleo de convolución.

    Returns:
        float: Suma de la multiplicación elemento a elemento.
    """
    return np.sum(fragment * kernel)


def convolution(image, kernel, padding=True):
    """
    Aplica una convolución con opción de padding a una imagen.

    Args:
        image (np.ndarray): Imagen en escala de grises.
        kernel (np.ndarray): Núcleo de convolución.
        padding (bool, optional): Si se aplica padding. Predeterminado es True.

    Returns:
        np.ndarray: Imagen resultante después de la convolución.
    """
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Calcula padding.
    pad_height = (kernel_row - 1) // 2
    pad_width = (kernel_col - 1) // 2

    # Aplica padding si es necesario.
    if padding:
        padded_image = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width))
        padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    else:
        padded_image = image

    output_row = image_row if padding else (image_row - kernel_row + 1)
    output_col = image_col if padding else (image_col - kernel_col + 1)
    output = np.zeros((output_row, output_col))

    # Aplica la convolución.
    for row in range(output_row):
        for col in range(output_col):
            fragment = padded_image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = conv_helper(fragment, kernel)

    return output


def normalize_image(image):
    """
    Normaliza la imagen para que los valores estén entre 0 y 255 y la convierte a uint8.

    Args:
        image (np.ndarray): Imagen a normalizar.

    Returns:
        np.ndarray: Imagen normalizada en tipo uint8.
    """
    try:
        # Reemplazar NaNs y Infs con cero
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Convertir a float32 si no lo es
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Verificar si la imagen tiene variación
        min_val = np.min(image)
        max_val = np.max(image)
        if min_val == max_val:
            # Evita la división por cero si la imagen tiene todos los mismos valores
            return np.zeros_like(image, dtype=np.uint8)

        # Normalizar la imagen al rango [0, 255]
        norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Convertir a uint8
        norm_image = norm_image.astype(np.uint8)

        return norm_image
    except cv2.error as e:
        print(f"Error en normalize_image: {e}")
        return np.zeros_like(image, dtype=np.uint8)
    except Exception as e:
        print(f"Error inesperado en normalize_image: {e}")
        return np.zeros_like(image, dtype=np.uint8)


def enhance_contrast(image):
    """
    Mejora el contraste usando ecualización del histograma.

    Args:
        image (np.ndarray): Imagen en escala de grises.

    Returns:
        np.ndarray: Imagen con contraste mejorado.
    """
    return cv2.equalizeHist(image)


def grayscale(image):
    """
    Convierte una imagen a escala de grises y aplica un filtro bilateral.

    Args:
        image (np.ndarray): Imagen en color.

    Returns:
        np.ndarray: Imagen en escala de grises con filtro bilateral aplicado.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray_image, 11, 17, 17)
    return bfilter


def decrease_contrast(image):
    """
    Disminuye el contraste de la imagen.

    Args:
        image (np.ndarray): Imagen normalizada.

    Returns:
        np.ndarray: Imagen con contraste reducido.
    """
    alpha = 0.15  # Factor de escala para el contraste.
    beta = 100  # Valor para ajustar el brillo.
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def adaptive_threshold_edges(image):
    """
    Aplica umbral adaptativo para binarizar la imagen.

    Args:
        image (np.ndarray): Imagen con contraste disminuido.

    Returns:
        np.ndarray: Imagen binarizada.
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 13, 2)


# Definir los filtros de convolución
filters = {
    'sobel_vertical': np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]),
    'sobel_horizontal': np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]),
    'laplacian': np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]]),
    'edge_detection': np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]]),
    'gaussian_blur': np.array([[0, 0, 0, 5, 0, 0, 0],
                               [0, 5, 18, 32, 18, 5, 0],
                               [0, 18, 64, 100, 64, 18, 0],
                               [5, 32, 100, 100, 100, 32, 5],
                               [0, 18, 64, 100, 64, 18, 0],
                               [0, 5, 18, 32, 18, 5, 0],
                               [0, 0, 0, 5, 0, 0, 0]])
}


def detectar_contorno_placa(imagen_gris, contornos, area_max=50):
    """
    Detecta el contorno de la placa de matrícula en una imagen en escala de grises.

    Args:
        imagen_gris (np.ndarray): Imagen en escala de grises.
        contornos (list): Lista de contornos detectados en la imagen.
        area_max (int, optional): Número máximo de contornos a considerar. Predeterminado es 50.

    Returns:
        np.ndarray or None: Coordenadas del contorno de la placa si se encuentra, de lo contrario None.
    """
    # Ordenar los contornos por área en orden descendente y considerar los primeros 'area_max'
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:area_max]

    # Inicializar la variable para almacenar el contorno de la placa
    contorno_placa = None

    # Iterar sobre los contornos para encontrar un cuadrilátero que represente la placa
    for contorno in contornos:
        # Aproximar la forma del contorno
        perimetro = cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

        # Si el contorno aproximado tiene cuatro puntos, asumir que es la placa
        if len(aproximacion) == 4:
            contorno_placa = aproximacion
            break

    return contorno_placa


def extraer_region_placa(imagen, contorno):
    """
    Extrae la región de la placa de matrícula de la imagen original usando el contorno proporcionado.

    Args:
        imagen (np.ndarray): Imagen original en color.
        contorno (np.ndarray): Coordenadas del contorno de la placa.

    Returns:
        np.ndarray: Imagen recortada de la placa.
    """
    # Crear una máscara para el contorno de la placa
    mascara = np.zeros(imagen.shape[:2], np.uint8)
    cv2.drawContours(mascara, [contorno], 0, 255, -1)

    # Obtener el rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(contorno)

    # Recortar la región de la placa de la imagen original
    placa_recortada = imagen[y:y+h, x:x+w]

    # Eliminar el 15% superior e inferior para reducir texto no relevante
    altura = placa_recortada.shape[0]
    recorte_superior = int(0.15 * altura)
    recorte_inferior = int(0.85 * altura)
    placa_recortada = placa_recortada[recorte_superior:recorte_inferior, :]

    return placa_recortada


def preprocesar_imagen(placa, filters):
    """
    Preprocesa la imagen de la placa para mejorar la precisión del OCR.

    Args:
        placa (np.ndarray): Imagen recortada de la placa en color.
        filters (dict): Diccionario de filtros de convolución personalizados.

    Returns:
        np.ndarray: Imagen preprocesada en escala de grises y binarizada.
    """
    # Convertir a escala de grises
    placa_gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    plotImages(placa, placa_gris, 'Imagen Original', 'Imagen en Escala de Grises')

    # Aumentar el contraste
    placa_gris = cv2.convertScaleAbs(placa_gris, alpha=1.5, beta=0)
    plotImages(placa, placa_gris, 'Imagen en Escala de Grises', 'Imagen con Contraste Aumentado')

    # Aplicar filtro de Sobel personalizado para detección de bordes horizontales
    sobel_horizontal = filters['sobel_horizontal']
    placa_sobel_horizontal = convolution(placa_gris, sobel_horizontal, padding=True)
    placa_sobel_horizontal = normalize_image(placa_sobel_horizontal)
    print(
        f"Bordes Sobel Horizontales - min: {np.min(placa_sobel_horizontal)}, max: {np.max(placa_sobel_horizontal)}, mean: {np.mean(placa_sobel_horizontal)}")
    plotImages(placa, placa_sobel_horizontal, 'Imagen con Contraste Aumentado', 'Bordes Sobel Horizontales')

    # Aplicar filtro de Sobel personalizado para detección de bordes verticales
    sobel_vertical = filters['sobel_vertical']
    placa_sobel_vertical = convolution(placa_gris, sobel_vertical, padding=True)
    placa_sobel_vertical = normalize_image(placa_sobel_vertical)
    print(
        f"Bordes Sobel Verticales - min: {np.min(placa_sobel_vertical)}, max: {np.max(placa_sobel_vertical)}, mean: {np.mean(placa_sobel_vertical)}")
    plotImages(placa, placa_sobel_vertical, 'Bordes Sobel Horizontales', 'Bordes Sobel Verticales')

    # Combinar los bordes horizontales y verticales usando la magnitud del gradiente
    combined_edges = np.sqrt(placa_sobel_horizontal ** 2 + placa_sobel_vertical ** 2)
    combined_edges = normalize_image(combined_edges)
    print(
        f"Bordes Sobel Combinados - min: {np.min(combined_edges)}, max: {np.max(combined_edges)}, mean: {np.mean(combined_edges)}")
    plotImages(placa, combined_edges, 'Bordes Sobel Verticales', 'Bordes Sobel Combinados')

    # Verificar si combined_edges no está vacío antes de normalizar
    if combined_edges.size == 0:
        print("Advertencia: La imagen combinada de bordes está vacía.")
        return combined_edges

    # Aplicar desenfoque Gaussiano para reducir el ruido
    combined_edges = cv2.GaussianBlur(combined_edges, (3, 3), 0)
    plotImages(placa, combined_edges, 'Bordes Sobel Combinados', 'Bordes con Desenfoque Gaussiano')

    # Aplicar umbralización de Otsu para binarizar la imagen
    _, placa_umbral = cv2.threshold(
        combined_edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    plotImages(placa, placa_umbral, 'Bordes con Desenfoque Gaussiano', 'Imagen Binarizada (Umbral de Otsu)')

    # Aplicar apertura morfológica para eliminar pequeñas imperfecciones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    placa_umbral = cv2.morphologyEx(placa_umbral, cv2.MORPH_OPEN, kernel)
    plotImages(placa, placa_umbral, 'Imagen Binarizada (Umbral de Otsu)', 'Imagen Binarizada con Apertura Morfológica')

    # Redimensionar la imagen para mejorar la precisión del OCR
    placa_redimensionada = cv2.resize(placa_umbral, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    plotImages(placa, placa_redimensionada, 'Imagen Binarizada con Apertura Morfológica',
               'Imagen Redimensionada para OCR')

    return placa_redimensionada



def filtrar_texto_placa(texto):
    """
    Filtra y valida el texto de la placa para asegurarse de que cumple con el formato esperado.

    Args:
        texto (str): Texto detectado por OCR.

    Returns:
        str or None: Texto válido de la placa o None si no cumple con el formato.
    """
    # Definir un patrón esperado usando expresiones regulares
    # Por ejemplo, placas con 3 letras seguidas de 3 números
    import re
    patron = r'^[A-Z]{3}\s?[0-9]{3}$'
    if re.match(patron, texto.replace(' ', '')):
        return texto.replace(' ', '')
    else:
        print("Texto de la placa no cumple con el formato esperado.")
        return None


def obtener_texto_placa(placa_preprocesada, umbral_confianza=0.5):
    """
    Utiliza EasyOCR para extraer el texto de la placa preprocesada.

    Args:
        placa_preprocesada (np.ndarray): Imagen preprocesada de la placa.
        umbral_confianza (float, optional): Umbral de confianza para filtrar detecciones. Predeterminado es 0.5.

    Returns:
        str or None: Texto de la placa si se encuentra, de lo contrario None.
    """
    # Inicializar el lector de EasyOCR
    lector = easyocr.Reader(['en'], gpu=False)

    # Realizar la detección de texto
    resultado = lector.readtext(placa_preprocesada, detail=1, paragraph=False)

    if len(resultado) == 0:
        print("No se encontró texto mediante EasyOCR.")
        return None
    else:
        # Filtrar resultados con baja confianza
        resultados_filtrados = [res for res in resultado if res[2] > umbral_confianza]

        if len(resultados_filtrados) == 0:
            print("No se encontró texto con suficiente confianza.")
            return None
        else:
            # Ordenar las detecciones por posición vertical (de arriba a abajo)
            resultados_filtrados.sort(key=lambda x: x[0][0][1])

            # Combinar textos que están cerca verticalmente
            texto_placa = ''
            y_anterior = None
            for res in resultados_filtrados:
                bbox, texto, confianza = res
                texto = texto.upper()

                # Reemplazar caracteres comunes mal reconocidos
                texto = texto.replace('O', '0').replace('I', '1').replace('L', '1')
                texto = texto.replace('Z', '2').replace('S', '5').replace('B', '8')
                texto = texto.replace('G', '6')

                y_actual = bbox[0][1]
                if y_anterior is None or abs(y_actual - y_anterior) < 20:
                    texto_placa += texto + ' '
                else:
                    # Iniciar una nueva línea si el texto está lejos verticalmente
                    texto_placa = texto + ' '
                y_anterior = y_actual

            texto_placa = texto_placa.strip()
            print("Texto detectado en la placa:", texto_placa)

            texto_placa = obtener_texto_placa(placa_preprocesada, umbral_confianza=0.5)
            texto_placa = filtrar_texto_placa(texto_placa) if texto_placa else None

            return texto_placa


def sobreponer_texto(imagen, texto, contorno):
    """
    Sobrepone el texto detectado en la imagen original, centrado en la placa y resalta el contorno.

    Args:
        imagen (np.ndarray): Imagen original en color.
        texto (str): Texto de la placa detectado.
        contorno (np.ndarray): Coordenadas del contorno de la placa.

    Returns:
        np.ndarray: Imagen con el texto sobrepuesto y el contorno resaltado.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calcular el centro del contorno de la placa
    momentos = cv2.moments(contorno)
    if momentos['m00'] != 0:
        centro_x = int(momentos['m10'] / momentos['m00'])
        centro_y = int(momentos['m01'] / momentos['m00'])
    else:
        # Alternativa: usar el centro del rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contorno)
        centro_x = x + w // 2
        centro_y = y + h // 2

    # Obtener el tamaño del texto para centrarlo
    (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, fontFace=font, fontScale=1.2, thickness=3)

    # Calcular la posición del texto para que esté centrado
    pos_x = centro_x - ancho_texto // 2
    pos_y = centro_y + alto_texto // 2

    # Asegurar que el texto esté dentro de los límites de la imagen
    pos_x = max(0, min(pos_x, imagen.shape[1] - ancho_texto))
    pos_y = max(alto_texto, min(pos_y, imagen.shape[0] - 10))

    # Sobreponer el texto en la imagen
    cv2.putText(imagen, texto, org=(pos_x, pos_y), fontFace=font,
                fontScale=1.2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

    # Resaltar el contorno de la placa
    cv2.drawContours(imagen, [contorno], -1, (0, 255, 0), 3)

    return imagen


def mostrar_imagenes(original, preprocesada, final):
    """
    Muestra las imágenes original, preprocesada y final utilizando Matplotlib.

    Args:
        original (np.ndarray): Imagen original en color con el contorno de la placa.
        preprocesada (np.ndarray): Imagen preprocesada de la placa.
        final (np.ndarray): Imagen final con el texto sobrepuesto.
    """
    plt.figure(figsize=(18, 6))

    # Imagen Original con Contorno de la Placa
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original con Contorno de la Placa')
    plt.axis('off')

    # Imagen Preprocesada de la Placa
    plt.subplot(1, 3, 2)
    plt.imshow(preprocesada, cmap='gray')
    plt.title('Imagen Preprocesada de la Placa')
    plt.axis('off')

    # Imagen Final con Texto Sobrepuesto
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Final con Texto Sobrepuesto')
    plt.axis('off')

    plt.show()


def main():
    """
    Función principal para ejecutar el proceso de reconocimiento de placas.
    """
    # Ruta de la imagen
    ruta_imagen = './images/plate1.jpg'  # Reemplazar con la ruta correcta

    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen_original = imagen.copy()

    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    print(f"Imagen en escala de grises: shape={imagen_gris.shape}, dtype={imagen_gris.dtype}")
    plotImages(imagen, imagen_gris, 'Imagen Original', 'Imagen en Escala de Grises')

    # Aplicar filtro bilateral para reducir el ruido manteniendo los bordes
    filtro_bilateral = cv2.bilateralFilter(imagen_gris, 11, 17, 17)
    print(f"Imagen con filtro bilateral: shape={filtro_bilateral.shape}, dtype={filtro_bilateral.dtype}")
    plotImages(imagen, filtro_bilateral, 'Imagen Original', 'Imagen con Filtro Bilateral')

    # Detectar bordes utilizando el detector de Canny
    bordes = cv2.Canny(filtro_bilateral, 30, 200)
    print(f"Bordes detectados con Canny: shape={bordes.shape}, dtype={bordes.dtype}")
    plotImages(imagen, bordes, 'Imagen con Filtro Bilateral', 'Bordes Detectados con Canny')

    # Encontrar contornos en la imagen con bordes detectados
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Número de contornos detectados: {len(contornos)}")

    # Detectar el contorno de la placa
    contorno_placa = detectar_contorno_placa(imagen_gris, contornos, area_max=50)

    if contorno_placa is None:
        print("No se encontró el contorno de la placa.")
        return
    else:
        # Dibujar el contorno de la placa en la imagen original
        cv2.drawContours(imagen_original, [contorno_placa], -1, (0, 255, 0), 3)
        print("Contorno de la placa detectado y dibujado en la imagen original.")
        plotImages(imagen, imagen_original, 'Bordes Detectados con Canny', 'Contorno de la Placa Detectado')

        # Extraer la región de la placa
        placa_recortada = extraer_region_placa(imagen, contorno_placa)
        if placa_recortada.size == 0:
            print("Error: La región recortada de la placa está vacía.")
            return
        print(f"Región de la placa recortada: shape={placa_recortada.shape}, dtype={placa_recortada.dtype}")
        plotImages(imagen_original, placa_recortada, 'Contorno de la Placa Detectado', 'Región de la Placa Recortada')

        # Preprocesar la imagen de la placa
        placa_preprocesada = preprocesar_imagen(placa_recortada, filters)
        print(f"Imagen preprocesada de la placa: shape={placa_preprocesada.shape}, dtype={placa_preprocesada.dtype}")

        # Verificar si la imagen preprocesada está vacía
        if placa_preprocesada.size == 0:
            print("Error: La imagen preprocesada de la placa está vacía.")
            return

        # Obtener el texto de la placa utilizando OCR
        texto_placa = obtener_texto_placa(placa_preprocesada, umbral_confianza=0.5)

        if texto_placa:
            # Sobreponer el texto detectado en la imagen original
            imagen_final = sobreponer_texto(imagen_original, texto_placa, contorno_placa)
            print("Texto detectado y sobrepuesto en la imagen original.")

            # Mostrar las imágenes en diferentes etapas del proceso
            mostrar_imagenes(imagen_original, placa_preprocesada, imagen_final)
        else:
            print("No se pudo detectar el número de la placa.")



if __name__ == "__main__":
    main()
