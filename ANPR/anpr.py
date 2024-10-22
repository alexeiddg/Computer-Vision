import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr

def detectar_contorno_placa(imagen_gris, contornos, area_max=50):
    """
    Detecta el contorno de la placa de matrícula en una imagen en escala de grises.

    Args:
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

def preprocesar_imagen(placa):
    """
    Preprocesa la imagen de la placa para mejorar la precisión del OCR.

    Args:
        placa (np.ndarray): Imagen recortada de la placa en color.

    Returns:
        np.ndarray: Imagen preprocesada en escala de grises y binarizada.
    """
    # Convertir a escala de grises
    placa_gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    # Aumentar el contraste
    placa_gris = cv2.convertScaleAbs(placa_gris, alpha=1.5, beta=0)

    # Aplicar desenfoque Gaussiano para reducir el ruido
    placa_gris = cv2.GaussianBlur(placa_gris, (3, 3), 0)

    # Aplicar umbralización de Otsu para binarizar la imagen
    _, placa_umbral = cv2.threshold(
        placa_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Aplicar apertura morfológica para eliminar pequeñas imperfecciones
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    placa_umbral = cv2.morphologyEx(placa_umbral, cv2.MORPH_OPEN, kernel)

    # Redimensionar la imagen para mejorar la precisión del OCR
    placa_redimensionada = cv2.resize(placa_umbral, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return placa_redimensionada

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

    # Mostrar la imagen original con el contorno de la placa
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original con Contorno de la Placa')
    plt.axis('off')

    # Mostrar la imagen preprocesada
    plt.subplot(1, 3, 2)
    plt.imshow(preprocesada, cmap='gray')
    plt.title('Imagen Preprocesada de la Placa')
    plt.axis('off')

    # Mostrar la imagen final con el texto sobrepuesto
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Final con Texto sobrepuesto')
    plt.axis('off')

    plt.show(block=False)
    plt.show()
    plt.pause(0.1)

def main():
    """
    Función principal para ejecutar el proceso de reconocimiento de placas.
    """
    # Ruta de la imagen
    ruta_imagen = './images/placa.png'

    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen_original = imagen.copy()

    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para reducir el ruido manteniendo los bordes
    filtro_bilateral = cv2.bilateralFilter(imagen_gris, 11, 17, 17)

    # Detectar bordes utilizando el detector de Canny
    bordes = cv2.Canny(filtro_bilateral, 30, 200)

    # Encontrar contornos en la imagen con bordes detectados
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Detectar el contorno de la placa
    contorno_placa = detectar_contorno_placa(imagen_gris, contornos, area_max=50)

    if contorno_placa is None:
        print("No se encontró el contorno de la placa.")
        return
    else:
        # Extraer la región de la placa
        placa_recortada = extraer_region_placa(imagen, contorno_placa)

        # Preprocesar la imagen de la placa
        placa_preprocesada = preprocesar_imagen(placa_recortada)

        # Obtener el texto de la placa utilizando OCR
        texto_placa = obtener_texto_placa(placa_preprocesada, umbral_confianza=0.5)

        if texto_placa:
            # Sobreponer el texto detectado en la imagen original
            imagen_final = sobreponer_texto(imagen_original, texto_placa, contorno_placa)

            # Mostrar las imágenes en diferentes etapas del proceso
            mostrar_imagenes(imagen_original, placa_preprocesada, imagen_final)
        else:
            print("No se pudo detectar el número de la placa.")

if __name__ == "__main__":
    main()
