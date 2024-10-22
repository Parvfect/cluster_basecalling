
import argparse
import os
from training_loop_functions import prepare_data_for_training, train_model, test_model, set_variables
from datetime import datetime

parser = argparse.ArgumentParser(
                    prog='Motif Caller',
                    description='Training the motif caller',
                    epilog='Use parser to set some parameters for training')


dataset_path_default = 'empirical_train_dataset_v5_payload_seq.pkl'

parser.add_argument('--alpha', type=float, default=0.01, help='Parameter for ground truth loss')
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--dataset_path', type=str, default=dataset_path_default)
parser.add_argument('--sample_data', type=bool, default=False)
parser.add_argument('--cluster', type=bool, default=False)

args = parser.parse_args()

if __name__ == '__main__':
    alpha = args.alpha
    epochs = args.epochs
    dataset_path = args.dataset_path
    sample = args.sample_data
    cluster = args.cluster

    if cluster:
        uid = str(datetime.datetime.now()).replace(' ', '.').replace('-','').replace(':',"")
        uid += f'-alpha_{alpha}_epochs_{epochs}'

        savepath = os.path.join(os.environ['HOME'], os.path.join("training_logs", f"{uid}"))
        os.mkdir(savepath)

        model_save_path = os.path.join(savepath, "model.pth")
        file_write_path = os.path.join(savepath, "log.txt")
        test_data_path = os.path.join(os.environ['HOME'], "sampled_test_dataset_v4_spacers.pkl")
        dataset_path = os.path.join(os.environ['HOME'], 'empirical_train_dataset_v5_payload_seq.pkl')
        model_path = ""
        saved_model = False
    else:
        model_save_path = 'model.pth'
        test_data_path = 'sampled_test_dataset_v4_spacers.pkl'
        model_path = 'model.pth'
        saved_model = False
        file_write_path = 'training_logs.txt'


    set_variables(model_save_path,
        test_data_path,
        model_path,
        saved_model,
        file_write_path)

    X_train, X_test, X_val, y_train, y_test, y_train, y_val, payload_train, payload_test, test_X, test_y, test_payload, payload_val, model, optimizer = prepare_data_for_training(dataset_path, sample)

    model = train_model(
        X_train, X_val, y_train, y_val, payload_train, payload_val, test_X, test_y, test_payload, epochs, model,
        optimizer, alpha)
    
    test_model(model, X_test, y_test, payload_test, test_X, test_y, test_payload, alpha)


