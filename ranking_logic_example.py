import os
import subprocess
import time
import logging.handlers
# import multiprocessing


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Function '{func.__name__}' took:")
        print(f"Seconds: {elapsed_time:.2f}")
        print(f"Minutes: {elapsed_time / 60:.2f}")
        print(f"Hours: {elapsed_time / 3600:.2f}")

        return result

    return wrapper


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)

    out, err = p.communicate()
    return out


@measure_time
def build_index(filename, currentTime, baseDir):
    """
    Parse the trectext file given, and create an index.
    """
    pathToFolder = baseDir + 'Collections/IndriIndices/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)

    INDRI_BUILD_INDEX = '/lv_local/home/user/indri/bin/IndriBuildIndex'
    CORPUS_PATH = filename
    CORPUS_CLASS = 'trectext'
    MEMORY = '1G'
    INDEX = pathToFolder + currentTime
    STEMMER = 'krovetz'
    if os.path.exists(INDEX):
        run_bash_command('rm -r ' + INDEX)
    command = INDRI_BUILD_INDEX + ' -corpus.path=' + CORPUS_PATH + ' -corpus.class=' + CORPUS_CLASS + ' -index=' + INDEX + ' -memory=' + MEMORY + ' -stemmer.name=' + STEMMER
    out = run_bash_command(command)
    return INDEX


@measure_time
def merge_indices(asrIndex, baseDir, currentTime):
    """
    Merge indices of ASR and ClueWeb09. If MergedIndx exists, it will be deleted.
    """

    INDRI_DUMP_INDEX = '/lv_local/home/user/indri/bin/dumpindex'
    CLUEWEB = f'/lv_local/home/user/cluewebindex'
    pathToFolder = baseDir + 'Collections/'
    MERGED_INDEX = pathToFolder + f'/mergedindex_{currentTime}'
    run_bash_command('rm -r ' + MERGED_INDEX)
    command = INDRI_DUMP_INDEX + ' ' + MERGED_INDEX + ' merge ' + CLUEWEB + ' ' + asrIndex
    out = run_bash_command(command)
    return MERGED_INDEX


@measure_time
def run_ranking_model(mergedIndex, workingSet, currentTime, baseDir):
    """
    workingSet - a file in trec format that dictates which population to work on
    format is: <qid> Q0 <docid> <rank> <score> <experiment name>\n - rank and score arguments can be filled
    arbitrarily they are simply for the desired format

    """
    pathToFolder = baseDir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = mergedIndex
    WORKING_SET_FILE = workingSet
    MODEL_FILE = '/lv_local/home/user/content_modification_code-master/rank_models/model_lambdatamart'
    QUERIES_FILE = f'/lv_local/home/user/content_modification_code-master/data/query_files/queries_{current_time}.xml'

    FEATURES_DIR = pathToFolder + '/Features/' + currentTime
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    FEATURES_FILE = 'features'

    if not os.listdir(FEATURES_DIR):

        command = baseDir + 'scripts/LTRFeatures ' + QUERIES_FILE + ' -stream=doc -index=' + INDEX + ' -repository=' + INDEX + ' -useWorkingSet=true -workingSetFile=' + WORKING_SET_FILE + ' -workingSetFormat=trec'
        print(command)
        out = run_bash_command(command)
        print(out)
        out = run_command('mv doc*_* ' + FEATURES_DIR)
        print(out)

        if any(file.startswith("doc") for file in os.listdir('/lv_local/home/user/content_modification_code-master')):
            command = f"find /lv_local/home/user/content_modification_code-master -maxdepth 1 -name 'doc*' -exec mv {{}} /lv_local/home/user/content_modification_code-master/Results/Features/{currentTime}/ \\;"
            output = run_bash_command(command)
            print(output.decode('utf-8'))
        else:
            print("No files starting with 'doc' found in the specified directory.")

    command = 'perl ' + baseDir + 'scripts/generate.pl ' + FEATURES_DIR + ' ' + WORKING_SET_FILE
    print(command)
    out = run_bash_command(command)
    print(out)
    command = '/lv_local/home/user/opt/java/jdk1.8.0/bin/java -jar ' + baseDir + 'scripts/RankLib.jar -load ' + MODEL_FILE + ' -rank ' + FEATURES_FILE + ' -score predictions.tmp'
    print(command)
    out = run_bash_command(command)
    print(out)
    command = 'cut -f3 predictions.tmp > predictions'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command('rm predictions.tmp')
    RANKED_LIST_DIR = pathToFolder + 'RankedLists/'
    if not os.path.exists(RANKED_LIST_DIR):
        os.makedirs(RANKED_LIST_DIR)
    PREDICTIONS_FILE = 'predictions'
    command = 'perl ' + baseDir + '/scripts/order.pl ' + RANKED_LIST_DIR + '/LambdaMART' + currentTime + ' ' + FEATURES_FILE + ' ' + PREDICTIONS_FILE
    print(command)
    out = run_bash_command(command)
    print(out)
    return RANKED_LIST_DIR + '/LambdaMART' + currentTime




def main_task(currentTime):
    logger.info("Starting...")
    baseDir = '/lv_local/home/user/content_modification_code-master/'
    documents = f'/lv_local/home/user/content_modification_code-master/trecs/bot_followup_{currentTime}.trectext'
    workingSet = f'/lv_local/home/user/content_modification_code-master/trecs/working_set_{currentTime}.trectext'

    mergedIndex = baseDir + 'Collections/' + f'/mergedindex_{currentTime}' # in case of running only the ranking model
    res = run_ranking_model(mergedIndex, workingSet, currentTime, baseDir)
    print("run_ranking_model done...")
    logger.info("run_ranking_model done...")

    print(res)
    logger.info(f'{res}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('ranking_logger')
    logger.setLevel(logging.DEBUG)

    current_time = "llama@50"
    print(f'Starting version {current_time}...')
    logger.info(f'Starting version {current_time}...')
    main_task(current_time)
