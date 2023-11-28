import numpy as np
from PIL import ImageTk, Image

# load model
from keras.models import load_model

cnn = load_model('cnn_model.h5')
cnn.summary()

# import an image
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

from keras.preprocessing import image
import pandas as pd

# dataset cifar100
dataset = pd.read_csv("cifar100_classes.csv")
classes = list(dataset.iloc[:, 0].values)
classes_a = list(dataset.iloc[:, 1].values)
super_classes = list(dataset.iloc[:20, -1].values)

# predict image
test_image = image.load_img(path=file_path, target_size=(32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
test_ind = [i for i, element in enumerate(list(result[0])) if element != 0]
ind = [i for i, element in enumerate(list(result[0])) if element != 0][0]
ind_tof = classes.index(f'{classes_a[ind]}')
ind_super = int(ind_tof / 5)

# display results
root.destroy()
window = tk.Tk()
class_cifar = tk.Label(window, text=f'{classes_a[ind]} ({super_classes[ind_super]})')
class_cifar.pack()
img = Image.open(file_path)
max_width = 500
pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])
img_display = ImageTk.PhotoImage(img.resize((pixels_x, pixels_y)))
image_disp = tk.Label(image=img_display)
image_disp.pack(fill="both", expand="yes")
window.mainloop()
