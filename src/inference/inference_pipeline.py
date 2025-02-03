from typing import Any, Dict
from src.inference.inferer import Inferer
from src.postprocess.saver import ResultsParser
from src.utils.sahi_usage import sahi_usage  # For SAHI-based inference
from src.utils.save_sahi_animation import save_sahi_animation  

def run_inference(config: Dict[str, Any]) -> None:
    inferer = Inferer(config)
    results = inferer.infer(persist=True)
    parser = ResultsParser(results=results, config=config)
    # For SAHI-based tasks, the Inferer.infer() already returns predictions.
    if config['model']['task'] == 'slice':
        # Pass the already-obtained SAHI results to a new CSV-saving method.
        parser.parse_and_save_slice(results)
        if config.get('save_animations', False):
            save_sahi_animation(config)
    else:
        parser.parse_and_save()
