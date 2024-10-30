import os,sys
path = os.path.abspath("src/utils")
sys.path.append(path)

import logging
from datetime import datetime
from read_files import read_yaml
from constant import CONFIG_FILE_PATH

class Logger():
    
    def __init__(self,training:bool=False,testing:bool=False):
        config = read_yaml(CONFIG_FILE_PATH,training=training,testing=testing)
        if training: self.logDirectory = config.logs.training
        elif testing: self.logDirectory = config.logs.testing
            
    def create_logger(self):
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        logger = logging.getLogger(f"logger_{timestamp}")
        logger.setLevel(logging.DEBUG)

        log_file = f"{self.logDirectory}/log_{timestamp}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        
        return logger

