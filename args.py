import argparse

def parser():
    parser = argparse.ArgumentParser(description="Training of the selected model")
    
    parser.add_argument('model', help="select model")
    parser.add_argument('--cpu', default="gpu")
    
    args = parser.parse_args()
    
    return args