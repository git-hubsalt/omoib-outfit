from src.datasets.processor import FashionInputProcessor, FashionImageProcessor
from transformers import AutoTokenizer, AutoImageProcessor
from src.models.embedder import KORCLIPEmbeddingModel
from src.models.recommender import RecommendationModel
import torch
import os


def load_model(args):
    try:

        image_processor_dir = os.path.join(args.model_dir, args.image_processor)
        text_tokenizer_dir = os.path.join(args.model_dir, args.text_tokenizer)
        print(f'>>>> load image processor from {image_processor_dir}')
        print(f'>>>> load text tokenizer from {text_tokenizer_dir}')
        image_processor = AutoImageProcessor.from_pretrained(image_processor_dir)
        text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_dir)

        input_processor = FashionInputProcessor(
            categories=args.categories,
            use_image=args.use_image,
            image_processor=image_processor, 
            use_text=args.use_text,
            text_tokenizer=text_tokenizer, 
            text_max_length=args.text_max_length, 
            text_padding='max_length', 
            text_truncation=True, 
            outfit_max_length=args.outfit_max_length
        )
        
        print(' >>>> loading Korean CLIP Embedding model for K-Fashion')
        embedding_model = KORCLIPEmbeddingModel(
            input_processor=input_processor,
            hidden=args.hidden,
            huggingface=None,
            normalize=args.normalize,
            linear_probing=True,
            args=args
        )

        print(' >>>> loading Recommendation model for K-Fashion')
        recommendation_model = RecommendationModel(
            embedding_model=embedding_model,
            ffn_hidden=args.ffn_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads
        )
        
        # 모델 파일 존재 확인
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")
            
        print(f' >>>> loading checkpoint from {args.model_path}')
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            print(f'Checkpoint loaded successfully. Keys: {checkpoint.keys()}')
        except Exception as e:
            raise Exception(f"Error loading checkpoint: {str(e)}")

        if 'state_dict' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'state_dict' key")
            
        state_dict = checkpoint['state_dict']
        print(f'State dict contains {len(state_dict)} keys')
        
        try:
            recommendation_model.load_state_dict(state_dict)
            print('State dict loaded successfully')
        except Exception as e:
            raise Exception(f"Error loading state dict: {str(e)}")
            
        print(f'[COMPLETE] Load from {args.model_path}')
        
        return recommendation_model, input_processor
        
    except Exception as e:
        print(f'Error in load_model: {str(e)}')
        print(f'Current working directory: {os.getcwd()}')
        print(f'Files in current directory: {os.listdir(".")}')
        if os.path.exists('model'):
            print(f'Files in model directory: {os.listdir("model")}')
        raise