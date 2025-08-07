import pygame
import numpy as np
import keras
from PIL import Image
from data_preprocessing import CLASS_NAMES

model = keras.models.load_model("quickdraw_cnn_model.h5")

pygame.init()
window_size = (500, 500)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Draw a Shape!")

canvas = pygame.Surface(window_size)
canvas.fill((0, 0, 0))

drawing = False
pen_radius = 8
last_pos = None
prediction_text = ""

clock = pygame.time.Clock()

def predict_image():
    pygame.image.save(canvas, "temp.png")
    img = Image.open("temp.png").convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array, verbose=0)[0]
    label = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return f"{label} ({confidence:.2f})"
    
def draw(surface, start, end, width):
    if start and end:
        pygame.draw.line(surface, (255, 255, 255), start, end, width)


running = True
while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

      
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                canvas.fill((0, 0, 0))
                prediction_text = ""
            elif event.key == pygame.K_ESCAPE:
                running = False

    if drawing:
        current_pos = pygame.mouse.get_pos()
        draw(canvas, last_pos, current_pos, pen_radius * 2)
        last_pos = current_pos

    prediction_text = predict_image()

    screen.blit(canvas, (0, 0))

    font = pygame.font.SysFont(None, 36)
    pred_surface = font.render("Guess: " + prediction_text, True, (255, 255, 255))
    screen.blit(pred_surface, (10, 10))

    text = pygame.font.SysFont(None, 20)
    text = text.render("Space: Clear  |  Esc: Quit", True, (180, 180, 180))
    screen.blit(text, (10, 50))

    pygame.display.flip()

pygame.quit()