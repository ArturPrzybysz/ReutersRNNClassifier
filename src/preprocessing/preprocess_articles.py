import pandas as pd
from string import digits
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def preprocess_articles(df: pd.DataFrame):
    df.body = df.body.str.lower()
    df.body = df.body.map(_remove_numbers)
    df.body = df.body.map(_remove_short)
    df.body = df.body.map(_lemme)

    print("x")
    return df


def _lemme(string: str):
    return lemmatizer.lemmatize(string)


def _remove_numbers(string: str):
    return string.translate(str.maketrans('', '', digits))


def _remove_short(string: str):
    words = re.split(r'[ \t\n"]+', string=string)
    text = ""
    for word in words:
        if len(word) > 2:
            text += word + " "
    return text
