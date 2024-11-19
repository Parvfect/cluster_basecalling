
import argparse
import os
from training_loop_functions import prepare_data_for_training, train_model, test_model, set_variables
from datetime import datetime

parser = argparse.ArgumentParser(
                    prog='Motif Caller',
                    description='Training the motif caller',
                    epilog='Use parser to set some parameters for training')


dataset_path_default = r"C:\Users\Parv\Doc\HelixWorks\Basecalling\cycle_dataset\short_read_dataset.pkl"

parser.add_argument('--alpha', type=float, default=0, help='Parameter for ground truth loss')
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--infere nce', type=bool, default=False)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--sample_data', action='store_true', help="Sample the data")
parser.add_argument('--cluster', action='store_true', help="Running on the cluster")
parser.add_argument('--payload_flag', action='store_true', help="Extract payload from dataset")
parser.add_argument('--unseen_data_flag', action='store_true', help="Test on unseen cycles")

parser.set_defaults(sample_data=False, cluster=False, payload_flag=False, unseen_data_flag=False)

args = parser.parse_args()

if __name__ == '__main__':
    alpha = args.alpha
    epochs = args.epochs
    dataset_path = args.dataset_path
    sample = args.sample_data
    cluster = args.cluster
    payload_flag = args.payload_flag
    unseen_data_flag = args.unseen_data_flag

    if cluster:
        uid = str(datetime.now()).replace(' ', '.').replace('-','').replace(':',"")
        uid += f'-alpha_{alpha}_epochs_{epochs}'

        savepath = os.path.join(os.environ['HOME'], os.path.join("training_logs", f"{uid}"))
        os.mkdir(savepath)

        model_save_path = os.path.join(savepath, "model.pth")
        file_write_path = os.path.join(savepath, "log.txt")
        test_data_path = os.path.join(os.environ['HOME'], "sampled_test_dataset_v4_spacers.pkl")
        
        if not dataset_path:
          dataset_path = os.path.join(os.environ['HOME'], 'empirical_train_dataset_v6.pkl')
        
        model_path = ""
        saved_model = False
    else:
        model_save_path = 'model.pth'
        test_data_path = 'sampled_test_dataset_v4_spacers.pkl'
        model_path = 'model_underfit.pth'
        saved_model = False
        file_write_path = 'training_logs.txt'


    set_variables(model_save_path,
        test_data_path,
        model_path,
        saved_model,
        file_write_path)

    if payload_flag and unseen_data_flag:
        X_train, X_test, X_val, y_train, y_test, y_train, y_val, payload_train, payload_test, payload_val, test_X, test_y, test_payload, model, optimizer = prepare_data_for_training(dataset_path, sample, payload_flag=payload_flag, unseen_data_flag=unseen_data_flag)

        model = train_model(
        X_train, X_val, y_train, y_val, epochs, model, optimizer, payload_train, payload_val, test_X, test_y, test_payload, alpha, payload_flag=True, unseen_data_flag=True)

        test_model(model, X_test, y_test, payload_test, test_X, test_y, test_payload, alpha, payload_flag=True, unseen_data_flag=True)
    
    elif payload_flag:
        X_train, X_test, X_val, y_train, y_test, y_train, y_val, payload_train, payload_test, payload_val, model, optimizer = prepare_data_for_training(dataset_path, sample, payload_flag=payload_flag)

        model = train_model(
        X_train, X_val, y_train, y_val, epochs, model, optimizer, payload_train, payload_test, payload_val, alpha=alpha, payload_flag=True)
        test_model(model, X_test, y_test, payload_test, alpha, payload_flag=True)

    elif unseen_data_flag:
        X_train, X_test, X_val, y_train, y_test, y_train, y_val, test_X, test_y, test_payload, model, optimizer = prepare_data_for_training(dataset_path, sample, payload_flag=payload_flag)
        model = train_model(
        X_train, X_val, y_train, y_val, epochs, model,
        optimizer, alpha,
        test_X=test_X, test_y=test_y, test_payload=test_payload)

        test_model(model, X_test, y_test, test_X=test_X, alpha=alpha)

    else:
        X_train, X_test, X_val, y_train, y_test, y_train, y_val, model, optimizer =  prepare_data_for_training(dataset_path, sample, payload_flag=payload_flag, unseen_data_flag=unseen_data_flag)

        model = train_model(
            X_train, X_val, y_train, y_val, epochs, model,
            optimizer, alpha)
    
        test_model(model, X_test, y_test, alpha)


