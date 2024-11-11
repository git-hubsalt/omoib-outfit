import torch
import itertools
import os
import requests
from PIL import Image
from io import BytesIO
import json
import threading
from functools import lru_cache

from src.datasets.kfashion import DatasetArguments, KFashionDataset
from src.models.embedder import KORCLIPEmbeddingModel
from src.datasets.processor import FashionInputProcessor
from src.models.load import load_model
from src.utils.utils import *
from model_args import Args


def model_fn(model_dir):
    print('-----------[START] Loading model for inference-----------')
    try:
        args = Args()
        args.model_dir = model_dir
        args.model_path = os.path.join(model_dir, 'model.pth')
        args.num_workers = 1
        args.inference_batch_size = 1
        
        model, input_processor = load_model(args)
        model.eval()
        print('-----------[END] Loading model -----------')
        
        return {'model': model, 'input_processor': input_processor, 'args': args}
    except Exception as e:
        print(f'Error in model_fn: {str(e)}')
        raise

def input_fn(request_body, request_content_type):
    print('-----------[START] Get input for inference-----------')
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        clothes = input_data['clothes']
        exclude = input_data.get('excludes', [])
        style = input_data['style']
        
        tops = [(item['id'], item['name'], item['url']) for item in clothes if item['type'] == '상의']
        bottoms = [(item['id'], item['name'], item['url']) for item in clothes if item['type'] == '하의']
        
        combinations = list(itertools.product(tops, bottoms))
        print('-----------[END] preprocessed input for inference-----------')
        
        return {
            'combinations': combinations,
            'exclude': exclude,
            'style': style
        }
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    print('-----------[START] Prediction-----------')
    model = model_artifacts['model']
    input_processor = model_artifacts['input_processor']
    args = model_artifacts['args']
    
    combinations = input_data['combinations']
    exclude = input_data['exclude']
    style = input_data['style']
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for top, bottom in combinations:
        outfit = {
            'category': ['상의', '하의'],
            'image': [top[2], bottom[2]],
            'text': [top[1], bottom[1]],
            'style': [style] * 2
        }
        
        cp_score = inference(outfit, model, args, input_processor, device)
        results.append([top[0], bottom[0], cp_score])
    
    results.sort(key=lambda x: x[2], reverse=True)
    recommendation = []
    
    for result in results:
        if result[:-1] not in exclude:
            recommendation = result
            break
            
    return recommendation

def output_fn(prediction, accept):
    if accept == 'application/json':
        response = {
            'statusCode': 200,
            'body': prediction
        }
        return json.dumps(response)
    raise ValueError(f"Unsupported accept type: {accept}")

def prepare_input(args, input_processor, categories, images, texts, styles):
    inputs = input_processor(
        category=categories,
        images=images,
        texts=texts,
        styles=styles,
        do_pad=True
    )
    return {k: v.unsqueeze(0) for k, v in inputs.items()}

def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def inference(outfit, model, args, input_processor, device):
    images = [load_image(img_path) for img_path in outfit['image']]
    
    inputs = prepare_input(args, input_processor, outfit['category'], images, outfit['text'], outfit['style'])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        item_embeddings = model.batch_encode(inputs)
        cp_score = model.get_score(item_embeddings)
    
    return cp_score.item()