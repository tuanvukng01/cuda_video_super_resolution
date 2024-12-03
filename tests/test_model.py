from models.esrgan_model import load_pretrained_model

def test_model_loading():
    model = load_pretrained_model()
    assert model is not None
    print("Test passed: Model loaded successfully.")