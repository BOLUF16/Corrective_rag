import logging
from pathlib import Path

current_working_directory = Path.cwd()

def setup_logger(logger_name: str, log_file:str, log_level:int = logging.INFO):

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file, encoding= 'utf-8')
    file_handler.setLevel(log_level)

    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)s')
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)

    return logger

def create_folder_and_log_file(folder_name:str, file_name:str):

    new_path = current_working_directory.joinpath(folder_name)

    new_path.mkdir(exist_ok=True)
    log_file_path = new_path.joinpath(file_name)

    log_file_path.touch()

folder_name = "logs"
log_files_to_create = ["system.log", "userops.log", "llmresponse.log"]
for file in log_files_to_create:
    create_folder_and_log_file(folder_name,file)

system_logger = setup_logger("systemlogger", f'{current_working_directory}/logs/system.log')
userops_logger = setup_logger("userlogger", f'{current_working_directory}/logs/userops.log')
llmresponse_logger = setup_logger("llmresponselogger", f'{current_working_directory}/logs/llmresponse.log')
