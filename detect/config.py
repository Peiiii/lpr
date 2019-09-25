# coding:utf-8

DETECT_ROOT='detect'

# yolo
TRAIN_INPUT_SIZES = [320, 352, 384, 416]
TEST_INPUT_SIZE = 384
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 128
LEARN_RATE_INIT = 1e-4 * BATCH_SIZE / 6
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = 80

GT_PER_GRID = 3

# test
SCORE_THRESHOLD = 0.50    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.1     # The threshold of the IOU when implement NMS

# name and path
DATASET_PATH = '/Users/mac/game/VOC2007'
PROJECT_PATH = '/Users/mac/game'
WEIGHTS_DIR = DETECT_ROOT+'/weights'
WEIGHTS_PATH=WEIGHTS_DIR+'/yolo.ckpt-step-18000'
LOG_DIR = 'log'
CLASSES = ['License-plate']

