import torch
import itertools
import os
import requests
from PIL import Image
from io import BytesIO
import json
import threading
from functools import lru_cache
import boto3

from src.datasets.kfashion import DatasetArguments, KFashionDataset
from src.models.embedder import KORCLIPEmbeddingModel
from src.datasets.processor import FashionInputProcessor
from src.models.load import load_model
from src.utils.utils import *
from model_args import Args

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
QUEUE_URL = 'https://sqs.ap-northeast-2.amazonaws.com/565393031158/omoib-recommendation-queue'

sqs = boto3.client(
    "sqs",
    region_name="ap-northeast-2",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def send_sqs(username, timestamp, prediction):
    message_body = {
        "userId": username,
        "initial_timestamp": timestamp,
        "prediction": prediction
    }

    try:
        response = sqs.send_message(
            QueueUrl=QUEUE_URL, MessageBody=json.dumps(message_body)
        )
        print(f"Message sent to SQS with MessageId: {response['MessageId']}")

        return {"statusCode": 200, "body": json.dumps("Message sent successfully!")}
    except Exception as e:
        print(f"Error sending message: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps("Error sending message to SQS"),
        }


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
        
        userid = input_data['userId']
        timestamp = input_data['timestamp']

        required = input_data.get('requiredClothes', [])
        clothes = input_data['clothesList']
        exclude = input_data.get('exclude', [])
        style = input_data['style']
        
        tops = [(item['id'], item['name'], item['url'], item['type']) for item in clothes if item['type'] == '상의']
        bottoms = [(item['id'], item['name'], item['url'], item['type']) for item in clothes if item['type'] == '하의']
        hats = [(item['id'], item['name'], item['url'], item['type']) for item in clothes if item['type'] == '모자']
        shoes = [(item['id'], item['name'], item['url'], item['type']) for item in clothes if item['type'] == '신발']

        if len(required) > 0:
        # required item 포함해서 combination 만들어야 함, 일단 필수 포함 item은 각 카테고리별로 하나만 들어온다고 가정하고 구현
            for r_item in required:
                if (r_item['type'] == '상의'):
                    tops = [r_item]
                    continue
                if (r_item['type'] == '하의'):
                    bottoms = [r_item]
                    continue
                if (r_item['type'] == '모자'):
                    hats = [r_item]
                    continue
                if (r_item['type'] == '신발'):
                    shoes = [r_item]
                    continue
        
        result = list(itertools.product(tops, bottoms))

        combinations = []
        for top, bottom in result:

            combinations.append((top, bottom))

            for shoe in shoes:
                combinations.append((top, bottom, shoe))

            for hat in hats:
                combinations.append((top, bottom, hat))

            for shoe, hat in itertools.product(shoes, hats):
                combinations.append((top, bottom, shoe, hat))

        print('-----------[END] preprocessed input for inference-----------')
        
        return {
            'userid': userid,
            'timestamp': timestamp,
            'required': required,
            'combinations': combinations,
            'exclude': exclude,
            'style': style
        }
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    print('-----------[START] Prediction-----------')
    userid = input_data['userid']
    timestamp = input_data['timestamp']

    model = model_artifacts['model']
    input_processor = model_artifacts['input_processor']
    args = model_artifacts['args']
    
    combinations = input_data['combinations']
    exclude = input_data['exclude']
    style = input_data['style']
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for items in combinations:

        if len(items) == 2:
            top, bottom = items
            outfit = {
                'category': [top[3], bottom[3]],
                'image': [top[2], bottom[2]],
                'text': [top[1], bottom[1]],
                'style': [style] * 2
            }
        
        if len(items) == 3:
            top, bottom, other = items
            outfit = {
                'category': [top[3], bottom[3], other[3]],
                'image': [top[2], bottom[2], other[2]],
                'text': [top[1], bottom[1], other[1]],
                'style': [style] * 3
            }
        
        if len(items) == 4:
            top, bottom, other1, other2 = items
            outfit = {
                'category': [top[3], bottom[3], other1[3], other2[3]],
                'image': [top[2], bottom[2], other1[2], other2[3]],
                'text': [top[1], bottom[1], other1[1], other2[3]],
                'style': [style] * 4
            }

        cp_score = inference(outfit, model, args, input_processor, device)
        results.append([top[0], bottom[0], cp_score])
    
    results.sort(key=lambda x: x[2], reverse=True)
    recommendation = []
    
    for result in results:
        if result[:-1] not in exclude:
            recommendation = result
            break
    
    print('-----------[END] Prediction-----------')
    return (userid, timestamp, recommendation)

def output_fn(prediction, accept):
    userid, timestamp, recommendation = prediction
    send_sqs(userid, timestamp, recommendation)
    response = {
        'body': recommendation
    }
    return json.dumps(response)

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