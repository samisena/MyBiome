import os 
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from src.data.probiotic_analyzer import ProbioticAnalyzer
from src.data.database_manager import DatabaseManager
from src.data.probiotic_analyzer import ProbioticAnalyzer, LLMConfig


#* Sets up file logging - the console logs will be saved to a file 
logging.basicConfig(
    level=logging.INFO,  #info level (shows basic logs)    
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  #log shows time, current module name
                                #log level (info, error, etc) and the actual log message 
    handlers = [
        logging.FileHandler('llm_pipeline_test.log'),  #?will write the logs to a file 'llm_pipeline_test.log'
        logging.StreamHandler() #will also write the log to the console
    ]
)

#* Creates a logger object with the name of the current module / will show this module's name at the
#* start of logs
logger = logging.getLogger(__name__)
