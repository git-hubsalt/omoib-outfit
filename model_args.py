from dataclasses import dataclass

@ dataclass
class Args:
    
    # Dataset & Input Processor Settings
    polyvore_split = 'nondisjoint'
    categories = ['상의', '하의']
    outfit_max_length = 8
    use_image = True
    use_text = True
    text_max_length = 64

    # Embedder&Recommender Model Settings
    use_clip_embedding = False
    image_processor = 'code/huggingface/image_processor'
    text_tokenizer = 'code/huggingface/text_tokenizer'
    image_encoder = 'code/huggingface/image_encoder'
    text_encoder = 'code/huggingface/text_encoder'
    clip_huggingface = 'patrickjohncyh/fashion-clip'
    kor_huggingface = 'Bingsu/clip-vit-large-patch14-ko'
    huggingface = 'sentence-transformers/all-MiniLM-L12-v2'
    hidden = 128
    ffn_hidden = 2024
    n_layers = 6
    n_heads = 16
    normalize = True

    @property
    def load_model(self):
        return True if self.model_path is not None else False