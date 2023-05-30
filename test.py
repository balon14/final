from streamlit_punctuate import lang_detection


def test_lang_detection():
    assert lang_detection("Dmitry") == \
        "Язык введенного текста не русский! Попробуйте ввести текст еще раз."
    assert lang_detection(",,,,....") == \
        "Невозможно определить язык! Попробуйте ввести текст еще раз."
