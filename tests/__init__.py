import os
import sys
PROJECT_PATH = os.getcwd()
# fix paths so it works both from "python3 -m unittest" and vscode
sys.path.append(os.path.join(PROJECT_PATH,"tests"))
sys.path.append(os.path.join(PROJECT_PATH,"klongpy"))
