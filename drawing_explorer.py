import sys
from PyQt6.QtWidgets import (QApplication, QSlider, QWidget, QMainWindow, QVBoxLayout, 
                             QFileDialog, QMenuBar, QLabel, QHBoxLayout)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from collections import namedtuple
import colorsys

from utility_io import load_sketch_polyline_data


Vector2 = namedtuple('Vector2', ['x', 'y'])

NB_IMAGES = 100

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.9) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in RGB_tuples]

class SketchImage:
    def __init__(self, filename, scale=1):
        V, E, P = load_sketch_polyline_data( filename )

        self.strokes = [[V[i] for i in line] for line in P]
        
        self.width = 960
        self.height = 960

        self.nbstrokes = len(self.strokes)
        self.points = V

        # Generate colors for each stroke
        self.stroke_colors = generate_distinct_colors(self.nbstrokes)

        # Rotation logic (unchanged)
        theta = 45 / 180 * np.pi
        phi = 30 / 180 * np.pi

        rotate_x_matrix = np.asarray([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        rotate_y_matrix = np.asarray([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

        self.rotation_matrix = rotate_x_matrix @ rotate_y_matrix

        points = self.points @ self.rotation_matrix.T

        self.xmin, self.ymin, self.zmin = np.min(points, axis=0)
        self.xmax, self.ymax, self.zmax = np.max(points, axis=0)

        self.index = np.linspace(0, self.nbstrokes-1, NB_IMAGES-1, dtype=int)

        self.im = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 255))
        self.images = [self.im]

    def draw_stroke(self, i):
        add = self.stroke_image(i).transpose(Image.FLIP_TOP_BOTTOM)
        self.im = Image.alpha_composite(self.im, add)

        if i in self.index:
            self.images.append(self.im.filter(ImageFilter.SMOOTH))

        sys.stdout.write(f"\rStroke {i+1}/{self.nbstrokes}")
        sys.stdout.flush()

    def stroke_image(self, i):
        add = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(add)

        color = self.stroke_colors[i]  # Get the color for this stroke

        for j in range(len(self.strokes[i]) - 1):
            p0, p1 = self.strokes[i][j], self.strokes[i][j+1]
            p0 = self.canvasToScreen(p0)
            p1 = self.canvasToScreen(p1)
            draw.line((p0, p1), fill=color, width=3)

        return add

    def canvasToScreen(self, v):
        rotated_vec = self.rotation_matrix @ v

        x, y, z = rotated_vec

        width = (x - self.xmin) / (self.xmax - self.xmin) * 1/2 * self.width + 1/4 * self.width 
        height = (y - self.ymin) / (self.ymax - self.ymin) * 3/4 * self.height + 1/8 * self.height
        return Vector2(width, height)

    def draw_all_strokes(self):
        for i in range(len(self.strokes)):
            self.draw_stroke(i)
        print()

class GUI(QMainWindow):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Drawing Explorer')
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.slider)
        
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas)
        
        self.create_menu()
        
    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        
        open_action = file_menu.addAction('Open File')
        open_action.triggered.connect(self.open_file)
        
        save_action = file_menu.addAction('Save Image')
        save_action.triggered.connect(self.save_img)
        
        quit_action = file_menu.addAction('Quit')
        quit_action.triggered.connect(QApplication.instance().quit)
        
    def set_drawing(self, img):
        self.drawing = img
        self.slider.setRange(0, len(img.images) - 1)
        self.slider.setValue(len(img.images) - 1)
        self.slider.valueChanged.connect(self.move_slider)
        self.move_slider(len(img.images) - 1)
        
        # for i, image in enumerate(img.images):
        #     image.save('figs_normal/drawing_sequence/dataset_t2f_blender/' + str(i) + '.png')
        
        
    def move_slider(self, val):
        im = self.drawing.images[val].resize((self.canvas.width(), self.canvas.height()))
        self.set_canvas_image(im)
        
    def set_canvas_image(self, img):
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self.canvas.setPixmap(pixmap.scaled(self.canvas.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'drawing'):
            self.move_slider(self.slider.value())
            
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Drawing file ?", "", "obj Files (*.obj)")
        if file:
            im = SketchImage(file, self.scaling)
            im.draw_all_strokes()
            self.set_drawing(im)
        else:
            # If no file is selected, close the application
            self.close()
            
    def save_img(self):
        if hasattr(self, 'drawing'):
            file, _ = QFileDialog.getSaveFileName(self, "Save as...", f"image_{self.slider.value()}.png", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file:
                self.drawing.images[self.slider.value()].save(file)
                print(f'Saving file {file}')
                
if __name__ == '__main__':
    scale = 1
    if len(sys.argv) == 2:
        try:
            scale = float(sys.argv[1])
        except ValueError:
            print("Invalid scale value. Using default scale = 1.")
    else:
        print("Using default scale = 1.")
        
    app = QApplication(sys.argv)
    gui = GUI(scale)
    gui.show()
    
    # Immediately prompt for file open when the application starts
    gui.open_file()
    
    # Only enter the main event loop if a file was successfully opened
    if not gui.isHidden():
        sys.exit(app.exec())