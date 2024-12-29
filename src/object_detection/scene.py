class Scene:
    """contains a list of components which are sequentially in the same layout type"""

    def __init__(self, x1,y1,x2,y2, components, zoom_factor=1):
        self.components = components
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.zoom_factor = zoom_factor
        self.content = ""
    def get_coordinates(self):
        return [self.x1,self.y1,self.x2,self.y2]

    def __str__(self):
        return f"components:{self.components}, zoom_factor:{self.zoom_factor}, " \
               f"coordinates: x1:{self.x1}, x2:{self.x2}, y1:{self.y1}, y2:{self.y2}, content:{self.content}"

    def __repr__(self):
        return str(self)