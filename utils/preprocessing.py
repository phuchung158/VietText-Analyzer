import re
from pyvi import ViTokenizer

def normalize_text(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Xóa URL
    text = re.sub(r"[@#]\S+", "", text)  # Xóa Mention, hashtag
    
    # Giữ lại các ký tự chữ cái, số, khoảng trắng (xóa dấu câu chấm, phẩy, chấm hỏi...)
    text = re.sub(r"[^\w\s]", " ", text)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_pipeline(text):
    # 1. Làm sạch text nền
    text = normalize_text(text)
    if not text:
        return ""
        
    # 2. Chạy PyVi để tách từ ghép (Ví dụ: "thầy cô dạy rất hay" -> "thầy_cô dạy rất hay")
    tokenized_text = ViTokenizer.tokenize(text)
    
    # Đảm bảo trả về một CHUỖI VĂN BẢN (String) thay vì một List
    return tokenized_text

def load_and_preprocess(texts):
    """Tiền xử lý cho một danh sách/batch văn bản."""
    return [preprocess_pipeline(text) for text in texts]
