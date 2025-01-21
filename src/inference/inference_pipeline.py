from typing import Any, Dict
from src.inference.inferer import Inferer
from src.postprocess.saver import ResultsParser

def run_inference(config):
    inferer = Inferer(config)
    results = inferer.infer()
    parser = ResultsParser(results=results, config=config)
    parser.parse_and_save()
