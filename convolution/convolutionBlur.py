import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image, kernel, padding=True):
    """Aplica una convolución con opción de padding."""
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    pad_height = (kernel_row - 1) // 2
    pad_width = (kernel_col - 1) // 2

    if padding:
        padded_image = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width))
        padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    else:
        padded_image = image

    output_row = image_row if padding else (image_row - kernel_row + 1)
    output_col = image_col if padding else (image_col - kernel_col + 1)
    output = np.zeros((output_row, output_col))

    for row in range(output_row):
        for col in range(output_col):
            fragment = padded_image[row:row + kernel_row, col:col + kernel_col]
            output[row, col] = np.sum(fragment * kernel)

    return output

def gaussian_blur(image):
    """Aplica un Gaussian Blur utilizando un kernel más grande y fuerte."""
    kernel = np.array([[1,  2,  3,  2,  1,  0,  0,  0, 0],
                       [2,  4,  6,  4,  2,  0,  0,  0, 0],
                       [3,  6,  9,  6,  3,  0,  0,  0, 0],
                       [2,  4,  6,  4,  2,  0,  0,  0, 0],
                       [1,  2,  3,  2,  1,  0,  0,  0, 0],
                       [0,  0,  0,  0,  0,  0,  0,  0, 0],
                       [0,  0,  0,  0,  0,  0,  0,  0, 0],
                       [0,  0,  0,  0,  0,  0,  0,  0, 0],
                       [0,  0,  0,  0,  0,  0,  0,  0, 0]])
    
    return convolution(image, kernel)

def reduce_contrast(image):
    """Reduce el contraste usando un kernel más grande."""
    # Kernel para normalización más fuerte
    kernel = np.array([[1, 1, 1, 1, 1], 
                       [1, 1, 1, 1, 1], 
                       [1, 1, 1, 1, 1], 
                       [1, 1, 1, 1, 1], 
                       [1, 1, 1, 1, 1]]) / 25  # Kernel de box blur 5x5
    return convolution(image, kernel)

def detect_edges(image):
    """Aplica un detector de bordes usando el operador Sobel."""
    sobel_kernel = np.array([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]])  # Kernel de Sobel para detección de bordes
    edges = convolution(image, sobel_kernel)
    return edges

def process_image(image_path):
    """Procesa la imagen para detectar números de placas."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # 1. Aplicar Gaussian Blur
    blurred_image = gaussian_blur(image)

    # 2. Reducir el contraste
    low_contrast_image = reduce_contrast(blurred_image)

    # 3. Detección de bordes
    edges = detect_edges(low_contrast_image)

    # Mostrar resultados
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_image, cmap='gray')
    plt.title('Imagen con Gaussian Blur')

    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Detección de Bordes')

    plt.show()

# Ejemplo de uso
image_path = r'C:\Users\omarp\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\Documents\Conv\Computer-Vision\convolution\images\placa.png'
process_image(image_path)
