import pygame

from canvas import draw_circle, draw_polygon


SOUND_VISUALS = {
    "Clapping": {
        "shape": "circle",
        "color": (255, 180, 50)
    },
    "Siren": {
        "shape": "triangle",
        "color": (255, 50, 80)
    },
    "Car": {
        "shape": "polygon",
        "color": (50, 150, 255)
    }
}


pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Spectra AI - Visual Test")


#Pretend this came from the classifier
sound_class = input("Enter sound (Car / Siren / Clapping): ")

visual = SOUND_VISUALS.get(sound_class)

if visual is None:
    print("Unknown sound")
else:
    print(f"Testing visual for: {sound_class}")


running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    if visual:

        shape = visual["shape"]
        color = visual["color"]

        if shape == "circle":

            draw_circle(
                screen,
                color,
                (400, 300),
                80
            )

        elif shape == "triangle":

            points = [
                (400, 200),
                (300, 400),
                (500, 400)
            ]

            draw_polygon(
                screen,
                color,
                points
            )

        elif shape == "polygon":

            points = [
                (300, 200),
                (500, 200),
                (500, 400),
                (300, 400)
            ]

            draw_polygon(
                screen,
                color,
                points
            )

    pygame.display.flip()


pygame.quit()
