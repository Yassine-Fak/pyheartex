"""
This module defines the settings shared by the worker and the server
"""
import os


REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
RQ_QUEUE_NAME = os.environ.get('RQ_QUEUE_NAME', 'model_server')
MODEL_DIR = os.environ.get('MODEL_DIR', '~/.heartex/models')

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
QUEUES = [RQ_QUEUE_NAME]
