import os

def relative_to_project(*components):
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..',
        *components))
