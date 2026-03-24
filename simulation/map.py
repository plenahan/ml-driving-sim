import numpy as np

class Map:
    def __init__(self, rectangles, path, start, direction):
        self.rectangles = rectangles
        self.path = path
        self.start = start
        self.direction = direction
        distance = 0.0
        for line in path:
            x1, y1, x2, y2 = line
            distance += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        self.path_length = distance

    def rectangles(self):
        return self.rectangles

    @staticmethod
    def map1():

        # Outer borders
        outer = [
            (0, -500, 1000, 100, 0),  # top
            (0,  500, 1000, 100, 0),  # bottom
            (500, 0, 100, 1000, 0),   # right
            (-500, 0, 100, 1000, 0)   # left
        ]

        # Track path (simplified loop)
        track = [
            # Rough outline
            ##########
            #        #
            # ###### #
            #   #### #
            ### #    #
            #   ######
            # ########
            #        #
            ######## #
            #        #
            ##########
            (-50, -300, 800, 100, 0),
            (50, -100, 800, 100, 0),
            (150, 0, 600, 100, 0),
            (-100, 100, 100, 100, 0),
            (-350, 100, 200, 100, 0),
            (50, 200, 400, 100, 0),
            (-180, 250, 300, 100, -25),
        ]

        path = [
            (-400, -400, 400, -400),
            (400, -400, 400, -200),
            (400, -200, -400, -200),
            (-400, -200, -400, 0),
            (-400, 0, -200, 0),
            (-200, 0, -200, 160),
            (-200, 160, -400, 250),
            (-400, 250, -325, 400),
            (-325, 400, 300, 300),
            (300, 300, 300, 100),
            (300, 100, 0, 100),
        ]

        return Map(outer + track, path, np.array([-400.0, -400.0]), np.array([1.0, 0.0]))
    
    @staticmethod
    def map2():
        # Outer borders
        outer = [
            (0, -500, 1000, 100, 0),  # top
            (0,  500, 1000, 100, 0),  # bottom
            (500, 0, 100, 1000, 0),   # right
            (-500, 0, 100, 1000, 0),   # left
            (0, 0, 100, 100, 15),
        ]
        return Map(outer, [], np.array([-400.0, -400.0]), np.array([1.0, 0.0]))