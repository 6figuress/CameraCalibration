import os



def getBaseFolder()->str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)