#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


print(tf.__version__)


# In[71]:


CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}
NUM_CLASSES = len(CLASS_MAP)
def mapper(val):
    return CLASS_MAP[val]


# In[39]:


IMG_SAVE_PATH = "image_data"


# In[40]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer


# In[119]:


dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        dataset.append([img, directory])


# In[120]:


data, labels = zip(*dataset)
labels = list(map(mapper, labels))


# In[121]:


labels = np_utils.to_categorical(labels)


# In[122]:


data = np.array(data)
labels = np.array(labels)


# In[123]:


labels[:5]


# In[124]:


from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[125]:


base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


# In[126]:


headModel = base_model.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name = 'flatten')(headModel)
headModel = Dense(128, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation = "softmax")(headModel)


# In[127]:


model = Model(inputs = base_model.input, outputs = headModel)


# In[128]:


for layer in base_model.layers:
    layer.trainable = False


# In[129]:


model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[130]:


labels[1]


# In[131]:


model.fit(data, labels, epochs=10)


# In[132]:


model.save("rock-paper-scissors-model.h5")


# In[151]:


CLASS_MAPP = {
    0:"rock",
    1:"paper",
    2:"scissors",
    3:"none"
}
def mapp(val):
    return CLASS_MAPP[val]


# In[158]:


image_pa = "154.jpg"
img_p = cv2.imread(image_pa)


# In[159]:


image = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))


# In[160]:


pred = model.predict(np.array([image]))
move_code = np.argmax(pred[0])
move_name = mapp(move_code)
print("Predicted: {}".format(move_name))


# In[ ]:




