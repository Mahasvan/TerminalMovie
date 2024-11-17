print("Importing RemBG. This may take a while...")
from rembg import remove 
from PIL import Image, ImageFilter

from colortrans import rgbcomponents2short

class ImageProcessor:
    def __init__(self, source_path, dest_path) -> None:
        self.source_path = source_path
        self.dest_path = dest_path

    def remove_background(self):
        img = Image.open(self.source_path)
        output_image = remove(img)
        output_image.save(self.dest_path)
        return output_image
    
    def blur_bg(self):
        img = Image.open(self.source_path)
        output_image = remove(img)
        black = Image.new("RGBA", img.size, (0, 0, 0, 255))
        alpha = output_image.split()[3]
        blackened_foreground = Image.composite(black, output_image, alpha)
        img.paste(blackened_foreground, (0, 0), blackened_foreground)
        blurred_image = img.filter(ImageFilter.GaussianBlur(20))
        blurred_image.paste(output_image, (0, 0), output_image)
        blurred_image.save(self.dest_path)
        return blurred_image

    def downsample(self, chunk_size=4):
        # maxpool 4x4 chunks of pixels
        image = Image.open(self.dest_path)
        width, height = image.size
        new_image = Image.new("RGB", (width//chunk_size, height//chunk_size))
        for x in range(0, width, chunk_size):
            for y in range(0, height, chunk_size):
                chunk = [image.getpixel((i, j)) for i in range(x, x+chunk_size) for j in range(y, y+chunk_size)]
                new_image.putpixel((x//chunk_size, y//chunk_size), max(chunk))
        new_image.save(self.dest_path)
        return new_image

    def convert_to_xterm(self):
        img = Image.open(self.dest_path)
        # todo: implement this
        color_matrix = []
        width, height = img.size
        for y in range(height):
            row = []
            for x in range(width):
                r,g,b = img.getpixel((x, y))
                code, _ = rgbcomponents2short(r,g,b)
                row.append(code)
            color_matrix.append(row)
        return color_matrix
    