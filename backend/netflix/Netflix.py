from backend.utility.get_data import Parser

class Netflix(object):
    def __init__(self):
        self.p = Parser()

    def netflix_data(self):
        return self.p.netflix_data