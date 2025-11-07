# Shim to expose DDQNAgent from the existing file with a non-standard filename
# This lets `from ddqn_agent import DDQNAgent` work without renaming files.
import importlib.util
import os

_this_dir = os.path.dirname(__file__)
_candidate = os.path.join(_this_dir, 'colley_25512144_ddqn(AGENT).py')

if not os.path.exists(_candidate):
    raise ImportError(f"Expected agent file not found: {_candidate}")

_spec = importlib.util.spec_from_file_location('colley_25512144_ddqn_AGENT', _candidate)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Export the DDQNAgent symbol for compatibility
try:
    DDQNAgent = _mod.DDQNAgent
except AttributeError:
    raise ImportError('DDQNAgent class not found in the agent file')

# Also re-export anything else the original module might expect
for name in getattr(_mod, '__all__', []):
    globals()[name] = getattr(_mod, name)
