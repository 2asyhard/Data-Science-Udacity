import config as cfg
import csv

def first_save(_time):
    f = open(f'./results_{_time}.csv', 'a', newline='')
    csvWriter = csv.writer(f)

    csvWriter.writerow(["cfg.MEMORY_SIZE", cfg.MEMORY_SIZE])
    csvWriter.writerow(["cfg.MAX_EPISODES", cfg.MAX_EPISODES])
    csvWriter.writerow(["cfg.MAX_STEPS", cfg.MAX_STEPS])
    csvWriter.writerow(["cfg.REWARD", cfg.REWARD])
    csvWriter.writerow(["cfg.GAMMA", cfg.GAMMA])
    csvWriter.writerow(["cfg.EPS_START", cfg.EPS_START])
    csvWriter.writerow(["cfg.EPS_END", cfg.EPS_END])
    csvWriter.writerow(["cfg.EPS_DECAY", cfg.EPS_DECAY])
    csvWriter.writerow(["cfg.TARGET_UPDATE", cfg.TARGET_UPDATE])
    csvWriter.writerow(["cfg.TRAIN_PERIOD", cfg.TRAIN_PERIOD])
    csvWriter.writerow(["cfg.MAX_TEST_STEP", cfg.MAX_TEST_STEP])
    csvWriter.writerow(["cfg.TEST_PERIOD", cfg.TEST_PERIOD])
    csvWriter.writerow(["cfg.FILTER_SIZE", cfg.FILTER_SIZE])
    csvWriter.writerow(["cfg.LEARNING_RATE", cfg.LEARNING_RATE])
    csvWriter.writerow(["cfg.MOMENTUM", cfg.MOMENTUM])
    csvWriter.writerow(["cfg.BATCH_SIZE", cfg.BATCH_SIZE])

    f.close()

def save_data(_time, i_episode, step):
    f = open(f'./results_{_time}.csv', 'a', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow([i_episode, step])
    f.close()


