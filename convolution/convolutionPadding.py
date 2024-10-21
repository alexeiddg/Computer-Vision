import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv_helper(fragment, kernel):
    """Multiplica dos matrices y devuelve su suma."""
    return np.sum(fragment * kernel)

def convolution(image, kernel, padding=True):
    """Aplica una convolución con opción de padding."""
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Calcula las dimensiones del padding.
    pad_height = (kernel_row - 1) // 2
    pad_width = (kernel_col - 1) // 2

    # Crea la imagen con padding.
    if padding:
        padded_image = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width))
        padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    else:
        padded_image = image

    # Calcula las dimensiones de la imagen de salida.
    output_row = image_row if padding else (image_row - kernel_row + 1)
    output_col = image_col if padding else (image_col - kernel_col + 1)
    output = np.zeros((output_row, output_col))

    # Aplica la convolución.
    for row in range(output_row):
        for col in range(output_col):
            fragment = padded_image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = conv_helper(fragment, kernel)

    return output

def apply_filter(image_path, filter_type):
    """Lee la imagen, aplica el filtro y muestra el resultado."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Definición de filtros.
    filters = {
        'sobel_vertical': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'sobel_horizontal': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),

        # Filtros personalizados
        'simple_box_blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        'gaussian_blur': np.array([[0, 0, 0, 5, 0, 0, 0],
                                   [0, 5, 18, 32, 18, 5, 0],
                                   [0, 18, 64, 100, 64, 18, 0],
                                   [5, 32, 100, 100, 100, 32, 5],
                                   [0, 18, 64, 100, 64, 18, 0],
                                   [0, 5, 18, 32, 18, 5, 0],
                                   [0, 0, 0, 5, 0, 0, 0]]),

        'line_detection_horizontal': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
        'line_detection_vertical': np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
        'line_detection_45': np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
        'line_detection_135': np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]),

        'edge_detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),

        'laplacian_diagonal': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),

        'laplacian_gaussian': np.array([[0, 0, -1, 0, 0],
                                        [0, -1, -2, -1, 0],
                                        [-1, -2, 16, -2, -1],
                                        [0, -1, -2, -1, 0],
                                        [0, 0, -1, 0, 0]]),
    }

    if filter_type not in filters:
        print(f"Error: Filtro '{filter_type}' no disponible.")
        return

    kernel = filters[filter_type]
    filtered_image = convolution(image, kernel, padding=True)

    # Muestra las imágenes.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Imagen Filtrada ({filter_type})')

    plt.show()

# Prueba del filtro.
rutaHouse = r'C:\Users\omarp\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Documents\Conv\Computer-Vision\convolution\images\house.jpg'
rutaPlaca = r'C:\Users\omarp\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Documents\Conv\Computer-Vision\convolution\images\placa.png'


apply_filter(rutaHouse, 'line_detection_horizontal')