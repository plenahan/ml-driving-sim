import math
import pygame
from simulation.car import Car
from simulation.map import Map


class Renderer:
    
    def __init__(self, cars, map=Map.map1(), screen_size=(600, 600),):
        pygame.init()
        pygame.display.set_caption("Driving Simulation")
        self.clock = pygame.time.Clock()
        self.screen_size = screen_size
        self.window = pygame.display.set_mode((screen_size[0], screen_size[1]))
        self.map = map
        self.cars = cars
        self.barrier_color = (128, 128, 128)
        self.car_color = (255, 0, 0)
        self.path_color = (0, 255, 0)
        self.rays = [(0,0,0,0)]

    
    def render(self):
        # Handle events so the window responds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.window.fill((0, 0, 0))

        self.window.fill((255, 255, 255))
        #for rectangle in self.map.rectangles:
            #self.rectangle(*rectangle, self.barrier_color)
        for line in self.map.path:
            self.line(*line, self.path_color)
        for ray in self.rays:
            self.line(*ray, (255, 0, 255))
        for car in self.cars:
            self.rectangle(car.position[0], car.position[1], *car.size, car.rotation, self.car_color)
        
        pygame.display.update()

    def rectangle(self, x, y, width, height, rotation, color):
        points = []
        x, y = self.to_screen(x, y)
        width, height = self.screen_scale(width, height)

        # The distance from the center of the rectangle to
        # one of the corners is the same for each corner.
        radius = math.sqrt((height / 2)**2 + (width / 2)**2)

        # Get the angle to one of the corners with respect
        # to the x-axis.
        angle = math.atan2(height / 2, width / 2)

        # Transform that angle to reach each corner of the rectangle.
        angles = [angle, -angle + math.pi, angle + math.pi, -angle]

        # Convert rotation from degrees to radians.
        rot_radians = (math.pi / 180) * rotation

        # Calculate the coordinates of each point.
        for angle in angles:
            y_offset = -1 * radius * math.sin(angle + rot_radians)
            x_offset = radius * math.cos(angle + rot_radians)
            points.append((x + x_offset, y + y_offset))

        pygame.draw.polygon(self.window, color, points)
    
    def line(self, x1, y1, x2, y2, color):
        pygame.draw.line(self.window, color, self.to_screen(x1, y1), self.to_screen(x2, y2))

    def to_screen(self, x, y):
        # Convert world coordinates to screen coordinates
        screen_x = int((x / 1000.0 + 0.5) * self.screen_size[0])
        screen_y = int((0.5 - y / 1000.0) * self.screen_size[1])
        return [screen_x, screen_y]
    
    def screen_scale(self, x, y):
        return int((x / 1000.0) * self.screen_size[0]), int((y / 1000.0) * self.screen_size[1])