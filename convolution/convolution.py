import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv_helper(fragment, kernel):
    """Multiplica dos matrices y devuelve su suma."""
    f_row, f_col = fragment.shape
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row, col] * kernel[row, col]
    return result

def convolution(image, kernel):
    """Aplica una convolución sin padding y devuelve la imagen filtrada."""
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros((image_row - kernel_row + 1, image_col - kernel_col + 1))

    # Aplica la convolución en cada posición posible.
    for row in range(image_row - kernel_row + 1):
        for col in range(image_col - kernel_col + 1):
            output[row, col] = conv_helper(
                image[row:row + kernel_row, col:col + kernel_col], kernel
            )

    return output

def apply_filter(image_path, filter_type):
    """Lee la imagen, aplica el filtro seleccionado y muestra el resultado."""
    # Carga la imagen en escala de grises.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Define diferentes filtros.
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

    # Verifica que el filtro elegido exista.
    if filter_type not in filters:
        print(f"Error: Filtro '{filter_type}' no disponible.")
        return

    # Aplica la convolución usando el filtro seleccionado.
    kernel = filters[filter_type]
    filtered_image = convolution(image, kernel)

    # Muestra la imagen original y la filtrada.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Imagen Filtrada ({filter_type})')

    plt.show()

# Ejemplo de uso:
# Cambia 'tu_imagen.jpg' por la ruta de tu imagen.
# apply_filter('images/house.jpg', 'sobel_x')

# Definición de la ruta de la imagen.
rutaHouse = r'C:\Users\omarp\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Documents\Conv\Computer-Vision\convolution\images\house.jpg'
rutaPlaca = r'C:\Users\omarp\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Documents\Conv\Computer-Vision\convolution\images\placa.png'

# Filtro a Aplicar
filtro = 'laplacian_gaussian'

# Aplicación de Filtros
apply_filter(rutaHouse, filtro)



