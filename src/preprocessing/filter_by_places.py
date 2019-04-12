import os
import json

from ReutersRNNClassifier.paths import ROOT
from ReutersRNNClassifier.src.config import legal_labels, label_name


def filter_by_places_and_save(data_dir: str, target_dir: str, label_name: str, legal_labels: [str]):
    directory = os.fsencode(data_dir)

    parsed_objects = {}
    for label in legal_labels:
        parsed_objects[label] = []

    for filename in os.listdir(directory):
        filename = os.fsdecode(filename)
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename)) as file:
                json_object = json.load(file)
                assert isinstance(json_object, list)

                for article in json_object:
                    if label_name in article.keys() and "body" in article.keys():
                        assert isinstance(article[label_name], list)
                        label = article[label_name][0]
                        if len(article[label_name]) is 1 and label in legal_labels:
                            stripped_object = strip_article(article)
                            parsed_objects[label].append(stripped_object)

    for label, articles in parsed_objects.items():
        with open(os.path.join(target_dir, label + '.json'), 'w') as outfile:
            json.dump(articles, outfile)


def strip_article(article):
    return {"places": article["places"][0], "body": article["body"]}


filter_by_places_and_save(os.path.join(ROOT, "data", "train_and_test_raw"),
                          os.path.join(ROOT, "data", "dataset"),
                          label_name,
                          legal_labels)
