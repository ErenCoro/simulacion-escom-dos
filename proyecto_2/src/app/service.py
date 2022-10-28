from fastapi import FastAPI
from .models import *
from pydantic import BaseModel




#definiendo nombre 
app = FastAPI()


class Link_perceptron(BaseModel):
    data_url: str
   
class Weight(BaseModel):
    model_weights: list
    input_data: list

class Link_pocket(BaseModel):
    data_url: str
    max_iters: int
    max_iters = 100





@app.post("/linear/pla/train")
async def link(data: Link_perceptron):
    url = data.data_url
    weight = perceptron_train(url, ['label'])
    return weight/0


@app.post("/linear/pla/predict")
async def perceptron(test: Weight):
    weight_perceptron = test.model_weights
    features_perceptron = test.input_data
    labels =  perceptron_and_pocket_test(features_perceptron, weight_perceptron)
    return labels/0


@app.post("/linear/pocket/train")
async def link(data: Link_pocket):
    url = data.data_url
    max_iters = data.max_iters
    weight_pocket = pocket_train(url, ['label'], max_iters = max_iters)
    return weight_pocket/0


@app.post("/linear/pocket/predict")
async def pocket(test: Weight):
    weight_pocket = test.model_weights
    features_pocket = test.input_data
    labels =  perceptron_and_pocket_test(features_pocket, weight_pocket)
    return labels/0


