
import argparse
from training_loop_functions import prepare_data_for_training, train_model, test_model

parser = argparse.ArgumentParser(
                    prog='MOtif Caller',
                    description='Training the motif caller',
                    epilog='Use parser to set some parameters for training')


dataset_path_default = 'empirical_train_dataset_v5_payload_seq.pkl'

parser.add_argument('--alpha', type=float, default=0.001, help='Parameter for ground truth loss')
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--dataset_path', type=str, default=dataset_path_default)
parser.add_argument('--sample_data', type=bool, default=False)


args = parser.parse_args()

if __name__ == '__main__':
    alpha = args.alpha
    epochs = args.epochs
    dataset_path = args.dataset_path
    sample = args.sample_data

    X_train, X_test, X_val, y_train, y_test, y_train, y_val, payload_train, payload_test, test_X, test_y, test_payload, payload_val, model, optimizer = prepare_data_for_training(dataset_path, sample)

    model = train_model(
        X_train, X_val, y_train, y_val, payload_train, payload_val, test_X, test_y, test_payload, epochs, model,
        optimizer, alpha)
    
    
    test_model(model, X_test, y_test, payload_test, test_X, test_y, test_payload, alpha)


