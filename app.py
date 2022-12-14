import os
import subprocess
import glob
import json
import mysql.connector
from flask import Flask, request, g, abort
from flask_cors import CORS
from main_utils import check_session, load_txt_arr, load_json_arr, process_kill
from labelling import Labeller
import datetime
import shutil
import logging
import logging.handlers


app = Flask(__name__)
CORS(app)


@app.before_request
def before_request():
    print(request.remote_addr)
    
    # logging handler
    log_level = logging.DEBUG

    for handler in app.logger.handlers:
        app.logger.removeHandler(handler)

    root = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(root, 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    log_file = os.path.join(logdir, 'app.log')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    
    fileMaxByte = 1024 * 1024 * 100
    # file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=fileMaxByte, backupCount=10) 
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    defaultFormatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    file_handler.setFormatter(defaultFormatter)
    stream_handler.setFormatter(defaultFormatter)

    app.logger.addHandler(stream_handler)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(log_level)


    if request.remote_addr not in [os.getenv("KESCO-DB-HOST"), os.getenv("KESCO-INFERENCE"), '127.0.0.1']:
        abort(403)  # Forbidden

    # g.TRAIN_PATH = os.path.join(os.getcwd(), "auto_train.py")
    g.FILE_STORAGE_PATH = os.getenv("KESCO-DATASET-PATH")
    g.TRAIN_RESULTS_PATH = os.path.join(g.FILE_STORAGE_PATH, "train_results")    
    g.db = mysql.connector.connect(host=os.getenv("KESCO-DB-HOST"), port=os.getenv("KESCO-DB-PORT"), database=os.getenv("KESCO-DB-NAME"), user=os.getenv("KESCO-DB-USER"),
                               password=os.getenv("KESCO-DB-PASSWORD"))

    # epochs
    g.epochs = '160' #200(orig) -> 160(1차 수정) -> 60(60은 일단 빨리 돌려볼라고)
    g.mask_iters = '20000'


@app.teardown_appcontext
def teardown_appcontext(response):
    g.db.close()


@app.route('/')
def testing():
    return 'This is kesco training server'


@app.route('/train', methods=["POST", "GET"])
def train():
    db = g.db
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # check training state
    sql_training_list = f"select weightid, trainingYN, pid from ln_weight_info where trainingYN='Y';"
    cursor.execute(sql_training_list)
    training_list = cursor.fetchall()
    if len(training_list) > 0:
        return "Training process is running"


    # get train arguments
    sql_pretrained_info = "select classificationfileurl, segmentationfileurl, createtm from ln_weight_info where confirmYN ='Y';"
    cursor.execute(sql_pretrained_info)
    pretrained_info = cursor.fetchall()
    print(pretrained_info)


    # default model weights
    clf_orig = os.path.join(g.FILE_STORAGE_PATH, "weights", "efficientnet_over_current_acc_98.55.pt")
    mask_orig = os.path.join(g.FILE_STORAGE_PATH, "weights", "mask_rcnn_26_AP_98.33.pt")

    try:
        clf_path = pretrained_info[0][0].replace(".weights", ".pt").replace("/", "\\")
        mask_path = pretrained_info[0][1].replace(".weights", ".pt").replace("/", "\\")
    except:
        clf_path = clf_orig
        mask_path = mask_orig

    
    if not os.path.isfile(clf_path):
        clf_path = clf_orig
    if not os.path.isfile(mask_path):
        mask_path = mask_orig

    
    sql_weight_info = f"select createtm, datapath from ln_weight_info where weightid={weight_id};"
    cursor.execute(sql_weight_info)
    weight_info = cursor.fetchall()
    create_time = weight_info[0][0].strftime('%Y-%m-%d_%Hh%Mm%Ss')
    data_path = weight_info[0][1]
    
    # create_time = pretrained_info[0][2].strftime('%Y-%m-%d_%Hh%Mm%Ss')
    # data_path = g.FILE_STORAGE_PATH

    train_file = os.path.join(os.getcwd(), "auto_train.py")

    print("weight_id :", weight_id)
    print("clf_model_path :", clf_path)
    print("mask_model_path :", mask_path)
    print("data_path :", data_path)
    print("create_time :", create_time)
    print("TRAIN_PATH :", train_file)

    # training process
    train_proc = subprocess.Popen(['python', train_file, "--file_storage_path", g.FILE_STORAGE_PATH, "--data_path", data_path,
                                    "--start_time", create_time, "--mask_path", mask_path, "--clf_path", "", # clf_path,
                                    "--mask_iters", g.mask_iters, "--epochs", g.epochs], shell=True)

    # train_proc = subprocess.Popen(['python', train_file, "--file_storage_path", g.FILE_STORAGE_PATH, "--data_path", data_path,
    #                             "--start_time", create_time, "--mask_path", mask_path, "--clf_path", clf_path,
    #                             "--mask_iters", "40", "--mask_checkpoint", "20", "--epochs", "5",
    #                             "--val_per_epochs", "3", "--test_per_epochs", "3", "--testing"], shell=True)

    train_pid = train_proc.pid

    sql_update_training = f"UPDATE ln_weight_info SET trainingYN='Y', pid={train_pid} WHERE weightid={weight_id};"
    cursor.execute(sql_update_training)
    db.commit()

    # db close
    cursor.close()
    db.close()

    train_proc.communicate()   # 432000 sec = 5 days

    check_training_weights = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time, "*.weights")

    if len(glob.glob(check_training_weights)) == 0:
        training_stop(session_id=session_id, weight_id=weight_id, training_error=True)
        shutil.rmtree(os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time))
        shutil.rmtree(os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time))
        
        # db close
        if db.is_connected():
            cursor.close()
            db.close()

        # raise ValueError("Training has stopped.")
        
        return "Training Error occured"
    
    training_stop(session_id=session_id, terminate_mode=True)

    # clf weight file path
    clf_folder = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time)
    clf_weight_path = glob.glob(os.path.join(clf_folder, "*.weights"))[-1].replace("\\", "/")
    accuracy = clf_weight_path.split("_")[-1][:-8]

    # mask weight file path
    mask_folder = os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time)
    mask_weight_path = glob.glob(os.path.join(mask_folder, "*.weights"))[-1].replace("\\", "/")

    # db connection
    g.db = mysql.connector.connect(host=os.getenv("KESCO-DB-HOST"), port=os.getenv("KESCO-DB-PORT"), database=os.getenv("KESCO-DB-NAME"), user=os.getenv("KESCO-DB-USER"),
                               password=os.getenv("KESCO-DB-PASSWORD"))
    db = g.db
    cursor = db.cursor()

    # Database Update
    sql_update_training = f"UPDATE ln_weight_info SET classificationfileurl = '{clf_weight_path}', " \
                          f"segmentationfileurl = '{mask_weight_path}', accuracy = '{accuracy}', trainingYN='N' " \
                          f"WHERE weightid={weight_id};"
    cursor.execute(sql_update_training)
    db.commit()

    cursor.close()

    return "Training is Done"


@app.route('/stop', methods=["POST", "GET"])
def training_stop(session_id='0', weight_id=-1, training_error=False, terminate_mode=False):

    # global db
    db = g.db

    if not db.is_connected():
        g.db = mysql.connector.connect(host=os.getenv("KESCO-DB-HOST"), port=os.getenv("KESCO-DB-PORT"), database=os.getenv("KESCO-DB-NAME"), user=os.getenv("KESCO-DB-USER"),
                               password=os.getenv("KESCO-DB-PASSWORD"))
        db = g.db
    
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        # weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    ### terminate train processes

    if training_error: #
        # Kill the specific training process (when training error occured)
        sql_training_proc = f"select createtm, pid from ln_weight_info where weightid={weight_id};"
        cursor.execute(sql_training_proc)
        training_list = cursor.fetchall()
        create_time = training_list[0][0].strftime('%Y-%m-%d_%Hh%Mm%Ss')
        train_pid = training_list[0][1]

        try:
            process_kill(train_pid)
        except:
            print(f"process ID has terminated.(pid={train_pid})")
            pass
            
        
        clf_folder = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time)
        mask_folder = os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time)

        try:
            shutil.rmtree(clf_folder)
            shutil.rmtree(mask_folder)
        except:
            pass

        sql_update_training = f"UPDATE ln_weight_info SET trainingYN='P' WHERE weightid={weight_id};"
        cursor.execute(sql_update_training)
        db.commit()

    else:
        # Kill all training processes
        sql_training_proc = f"select createtm, pid from ln_weight_info where trainingYN='Y';"
        cursor.execute(sql_training_proc)
        training_list = cursor.fetchall()

        for training in training_list:
            create_time = training[0].strftime('%Y-%m-%d_%Hh%Mm%Ss')
            train_pid = training[1]
            
            try:
                process_kill(train_pid)
            except:
                print(f"process ID has terminated.(pid={train_pid})")
                pass
        if terminate_mode:
            sql_update_training = f"UPDATE ln_weight_info SET trainingYN='N' WHERE trainingYN='Y';"
            cursor.execute(sql_update_training)
            db.commit()
        else:
            sql_update_training = f"UPDATE ln_weight_info SET trainingYN='P' WHERE trainingYN='Y';"
            cursor.execute(sql_update_training)
            db.commit()

    cursor.close()

    return "Stopped"


@app.route('/progress', methods=["POST", "GET"])
def check_progress():
    #global db
    db = g.db
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        # weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    # default percentage
    percentage = '0'


    # get training info
    sql_training_info = "select createtm from ln_weight_info where trainingYN='Y';"

    cursor.execute(sql_training_info)
    createtm_info = cursor.fetchall()
    if len(createtm_info) == 0:
        return percentage

    create_time = createtm_info[0][0].strftime('%Y-%m-%d_%Hh%Mm%Ss') #orig
    #create_time = 

    # clf weight file path
    clf_folder = os.path.join(g.TRAIN_RESULTS_PATH, "efficientnet", create_time)
    mask_folder = os.path.join(g.TRAIN_RESULTS_PATH, "mask_rcnn", create_time)

    
    clf_log = os.path.join(clf_folder, "train.log")
    mask_log = os.path.join(mask_folder, "metrics.json")

    # Find progress of training using training logs.
    
    if os.path.isdir(mask_folder):
        percentage = '1' # training is started
        
        if os.path.isfile(mask_log):
            
            if os.path.isfile(clf_log):
                logs = load_txt_arr(clf_log)
                epochs = [x.split(' ')[2] for x in logs if "Train Epoch" in x]
                latest_epoch = 0
                if len(epochs) != 0:
                    latest_epoch = int(epochs[-1])
                percentage = str(int(latest_epoch / int(g.epochs) * 100 * 0.5) + 50)  # total epochs = 200, +50% : after mask rcnn complete
                return percentage

            metrics = load_json_arr(mask_log)
            iters = [x['iteration'] for x in metrics if 'total_loss' in x]
            latest_iter = 0
            if len(iters) != 0:
                latest_iter = iters[-1]
            percentage = str(int(latest_iter / int(g.mask_iters) * 100 * 0.5) + 1)  # total iteration = 20000
            return percentage

        return percentage

    cursor.close()

    return percentage


@app.route('/autolabel', methods=["POST", "GET"])
def autolabel():
    # global db
    db = g.db
    cursor = db.cursor()

    if request.get_json(silent=True):
        json_data = request.get_json()
        session_id = json_data["sessionid"]
        weight_id = json_data["weightid"]
    else:
        return "Cannot read request"

    if not check_session(db=db, session_id=session_id):
        return "Cannot verify session id"

    auto_labeller = Labeller(db, weight_id, nms_cnt=3, num_classes=26, wire_class_num=25,
                 image_shape=(720,1280), epsilon_rate=0.003)

    # try:
    #     test_id = Labeller.test()
    #     return test_id

    try:
        auto_labeller.labelling()
    except:
        return "Error in autolabelling"

    print("Autolabelling has done")
    return "Autolabelling has done"



if __name__=="__main__":
    app.run(host='0.0.0.0', port=80)
    # serve(app, host='0.0.0.0', port=80, threads=4)
