from typing import Dict
from io import BytesIO
import numpy as np
from PIL import Image
from fastai.vision.all import *

def read_image(file: bytes) -> PILImage:
    img = Image.open(BytesIO(file))
    fastimg = PILImage.create(np.array(img.convert('RGB')))

    return fastimg

#Allows fastai to use albumentations out of the box
class AlbumentationsTransform (RandTransform):
    split_idx,order=None,2
    def __init__(self, train_aug, valid_aug): store_attr()
    
    def before_call(self, b, split_idx):
        self.idx = split_idx
    
    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

#predict function
def predict_fruit(image) -> Dict:
    path = Path()
    learn = load_learner(path/'models/fruit_model_v2.pkl')
    labels = learn.dls.vocab
    pred, pred_idx, probs = learn.predict(img)
    max_idx = probs.index(max(probs))
    return {
        "prediction": labels[max_idx],
        "probability": probs[max_idx],
    }

    

