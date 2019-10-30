import torch
import pickle
import numpy as np

def load_dataset(dataset_path):
    with open(dataset_path, "rb") as pkl:
        data = pickle.load(pkl)
    return data


def count_samples_teacher_forcing(tokenized_data, max_length):
    n_samples = 0
    for chunk in tokenized_data:
        l = len(chunk)
        if l < max_length:
            n_samples += 1
        else:
            n_samples += l - max_length
    return n_samples

def preprocess_dataset(tokenized_data, input_length, max_length, teacher_forcing=True):
    tensor_datasets = []
    if not teacher_forcing:
        n_samples = len(tokenized_data)
        input_ids = np.full((n_samples, input_length), fill_value=0, dtype=np.int64)
        lm_labels = np.full((n_samples, input_length), fill_value=-1, dtype=np.int64)
        for i, chunk in enumerate(tokenized_data):
            l = len(chunk)
            if l < max_length:
                input_ids[i, :l] = chunk[:l]
                ################# ??????
                lm_labels[i, :l] = chunk[:l]
            else:
                input_ids[i, :max_length] = chunk[:max_length]
                lm_labels[i, :max_length] = chunk[:max_length]
        all_inputs = (input_ids, lm_labels)
        tensor_datasets.append(torch.tensor(all_inputs[0]))
        tensor_datasets.append(torch.tensor(all_inputs[1]))
        return tensor_datasets
    else:
        n_samples_found = count_samples_teacher_forcing(tokenized_data, max_length)
        print("dataset contains {} samples with teacher forcing".format(n_samples_found))
        all_samples = []
        for i, chunk in enumerate(tokenized_data):
            l = len(chunk)
            if l < max_length:
                # input_ids[i, :l] = chunk[:l]
                # lm_labels[i, :l] = chunk[:l]
                all_samples.append(chunk)
            else:
                j = 0
                while (j <= l - 1 - max_length):
                    tmp = chunk[j: j + max_length]
                    # input_ids[i, :max_length] = tmp
                    # lm_labels[i, :max_length] = tmp
                    all_samples.append(tmp)
                    j += 1
        assert len(all_samples) == n_samples_found
        input_ids = np.full((n_samples_found, input_length), fill_value=0, dtype=np.int64)
        lm_labels = np.full((n_samples_found, input_length), fill_value=-1, dtype=np.int64)
        for k, sample in enumerate(all_samples):
            l = len(sample)
            input_ids[k, :l] = sample[:l]
            lm_labels[k, :l] = sample[:l]
        all_inputs = (input_ids, lm_labels)
        tensor_datasets.append(torch.tensor(all_inputs[0]))
        tensor_datasets.append(torch.tensor(all_inputs[1]))
        return tensor_datasets, n_samples_found