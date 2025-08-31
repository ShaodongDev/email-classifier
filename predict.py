from typing import Tuple, Optional
import pyautogui
import pytesseract

def set_tesseract(cmd_path: str) -> None:
    """Point pytesseract to Tesseract executable."""
    pytesseract.pytesseract.tesseract_cmd = cmd_path

def ocr_predict(
    region: Tuple[int, int, int, int],
    vectorizer,
    model,
    tesseract_cmd: Optional[str] = None,
    debug: bool = True,
) -> Tuple[str, str, str]:
    """
    Capture a screen region → OCR → vectorize → predict.
    
    Args:
        region: (x, y, width, height) rectangle to capture.
        vectorizer: fitted TfidfVectorizer.
        model: fitted MultiOutputClassifier.
        tesseract_cmd: optional path to tesseract.exe.
        debug: print OCR text and predictions if True.
    Returns:
        subject_text, main_category, sub_category
    """
    if tesseract_cmd:
        set_tesseract(tesseract_cmd)

    # Screenshot → OCR text 
    img = pyautogui.screenshot(region=region)
    subject_text = pytesseract.image_to_string(img).strip() # TODO cleaning if needed

    # Vectorize & predict
    X = vectorizer.transform([subject_text])
    pred = model.predict(X)
    main_cat, sub_cat = pred[0, 0], pred[0, 1]

    if debug:
        print(f"OCR Subject: {subject_text}")
        print(f"Predicted → Main: {main_cat} | Sub: {sub_cat}")

    return subject_text, main_cat, sub_cat