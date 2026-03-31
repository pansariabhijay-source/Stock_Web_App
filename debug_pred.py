import logging
logging.basicConfig(level=logging.WARNING)
from api.model_registry import ModelRegistry
import traceback

print("Loading registry...")
m = ModelRegistry()
try:
    print("Regime:", m.get_current_regime())
except Exception as e:
    print("Failed:")
    traceback.print_exc()
