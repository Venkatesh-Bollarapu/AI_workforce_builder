import os
from dotenv import load_dotenv

load_dotenv()

WORKSPACE = "./workspace"

MAX_WORKER_STEPS = 20
MAX_REPAIR_ATTEMPTS = 3
MAX_EVIDENCE_FILE_SIZE = 30_000

os.makedirs(WORKSPACE, exist_ok=True)