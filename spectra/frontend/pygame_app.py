import os
import sys
import pygame
from spectra.processing.categories import ESC50_CATEGORIES, SOUNDS_DICT, DEFAULT_CATEGORY

# Inicializar Pygame
pygame.init()

# Configuración de ventana
WIDTH, HEIGHT = 900, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spectra AI - Audio Visualizer MVP")
clock = pygame.time.Clock()

# Colores (Dark Studio Theme)
BG_COLOR = (18, 18, 24)
CARD_BG = (27, 27, 36)
BORDER_COLOR = (42, 42, 54)
CYAN = (0, 242, 254)
TEXT_COLOR = (242, 240, 234)

font_title = pygame.font.SysFont("Arial", 28, bold=True)
font_body = pygame.font.SysFont("Arial", 20)

# Cargar imágenes de categorías
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

CATEGORY_IMAGES = {}
categories_list = list(ESC50_CATEGORIES.keys()) + [DEFAULT_CATEGORY]

for cat in categories_list:
    img_name = f"{cat.lower()}.png"
    img_path = os.path.join(ASSETS_DIR, img_name)
    if os.path.exists(img_path):
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (200, 200))  # Ajustar tamaño
        CATEGORY_IMAGES[cat] = img
    else:
        CATEGORY_IMAGES[cat] = None  # Fallback si no existe la imagen

# Estado inicial simulado
current_category = "Animal"
current_sound = "dog"

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)

    # Título
    title = font_title.render("Spectra AI - Live Classification", True, CYAN)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 30))

    # Tarjeta de renderizado para la categoría
    card_rect = pygame.Rect(WIDTH // 2 - 150, 100, 300, 350)
    pygame.draw.rect(screen, CARD_BG, card_rect, border_radius=12)
    pygame.draw.rect(screen, BORDER_COLOR, card_rect, width=2, border_radius=12)

    # Renderizar imagen de la categoría actual
    current_img = CATEGORY_IMAGES.get(current_category)
    if current_img:
        screen.blit(current_img, (card_rect.x + 50, card_rect.y + 30))

    # Texto de Categoría y Sonido Detectado
    cat_text = font_title.render(f"Cat: {current_category}", True, TEXT_COLOR)
    sound_text = font_body.render(f"Detected: {current_sound}", True, CYAN)

    screen.blit(cat_text, (card_rect.x + (150 - cat_text.get_width() // 2), card_rect.y + 250))
    screen.blit(sound_text, (card_rect.x + (150 - sound_text.get_width() // 2), card_rect.y + 290))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
