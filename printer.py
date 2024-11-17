import json

class Printer:
    def __init__(self, color_matrix_path) -> None:
        self.color_matrix_path = color_matrix_path
    
    def print_xterm_line(self, xterm_line):
        prev = None
        for char in xterm_line:
            if char == "16": print("  ", end=""); continue
            if char != prev:
                print(f"\x1b[38;5;{char}m██", end="")
                prev = char
            else:
                print("██", end="")
        print("\x1b[0m")
    
    def print_matrix(self):
        with open(self.color_matrix_path, 'r') as f:
            color_matrix = json.load(f)
        for line in color_matrix:
            self.print_xterm_line(line)