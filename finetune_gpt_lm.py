import datetime
import os
import argparse
from tqdm import tqdm, trange
import random
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import CrossEntropyLoss

from utils.dataset import count_samples_teacher_forcing, preprocess_dataset, load_dataset

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer,
                                    WEIGHTS_NAME, CONFIG_NAME)

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=2)
    # cap_length = len([x for x in labels if x != -1])
    # score = (outputs[:cap_length + 3] == labels[:cap_length +3])
    score = (outputs == labels)
    return np.sum(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-medium',
                        help='pretrained model name', choices=['openai-gpt', 'gpt2', 'gpt2-medium']
                        )
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.", default=True)
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.", default=True)
    parser.add_argument("--do_save", action="store_true", help="Whether to save the model", default=True)
    parser.add_argument("--output_dir", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.",
                        default='logs')
    parser.add_argument("--run_parallel", action="store_true", help="Whether to run on multiple GPU's", default=False)
    parser.add_argument('--train_dataset', type=str, default='data/all_clauses.pkl')
    parser.add_argument('--eval_dataset', type=str, default='data/all_clauses.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_step', type=int, default=2)
    parser.add_argument('--save_train_logs', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-3)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--teacher_forcing', type=bool, default=True)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    #SETUP
    # Debugging
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if (torch.cuda.is_available() and args.run_parallel) else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # logs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    now = str(datetime.datetime.now()).replace(" ", "_")
    log_dir = args.model_name + "_" + now
    log_dir_steps = 'intermediate_model_steps'
    log_dir = os.path.join(args.output_dir, log_dir)
    log_dir_steps = os.path.join(log_dir, log_dir_steps)
    os.makedirs(log_dir)
    os.makedirs(log_dir_steps)

    args.output_dir = log_dir
    print("train and eval logs saved in {}".format(args.output_dir))

    # Load dataset
    dataset_path = args.train_dataset
    data = load_dataset(dataset_path)[:10]

    # BPE tokenizer and model
    if args.model_name == "openai-gpt":
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name)
        model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name)
    elif args.model_name == "gpt2" or args.model_name == "gpt2-medium":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # We want to return the hidden states
    model.config.output_hidden_states = True

    # Model config parameters
    max_length = model.config.n_positions
    num_features = model.config.n_embd
    vocab_size = model.config.vocab_size

    if args.run_parallel and n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    tokenized_data = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(chunk)) for chunk in data]

    # Train/eval split
    train_tokens, eval_tokens = train_test_split(tokenized_data, test_size=0.3, shuffle=True)
    input_length = max([len(t) for t in tokenized_data])
    input_length = min(input_length, max_length)

    tensors_dataset_train, n_samples_train = preprocess_dataset(train_tokens, input_length, max_length, teacher_forcing=True)
    tensors_dataset_eval, n_samples_eval = preprocess_dataset(eval_tokens, input_length, max_length, teacher_forcing=True)

    train_data = TensorDataset(tensors_dataset_train[0], tensors_dataset_train[1])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(tensors_dataset_eval[0], tensors_dataset_eval[1])
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ####### Training SETUP

    if args.do_train:
        for param in model.parameters():
            param.requires_grad = False

    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

    ####### Supervised Finetuning
    if args.run_parallel and n_gpu > 1:
        print("We will use {} GPU's ".format(n_gpu))
        dense_layer = torch.nn.DataParallel(torch.nn.Linear(num_features, vocab_size, bias=False))
    else:
        dense_layer = torch.nn.Linear(num_features, vocab_size, bias=False)
    dense_layer.to(device)

    #training parameters!
    num_warmup_steps = int(args.warmup_proportion*num_train_optimization_steps)
    trainable_params = dense_layer.parameters()
    #
    # optimizer_parameters = [{'params': list(trainable_params),
    #                          'weight_decay': 0.01}]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_warmup_steps,
                                     t_total=num_train_optimization_steps
                                     )

    if args.run_parallel and n_gpu > 1:
        loss_fct = torch.nn.DataParallel(CrossEntropyLoss(ignore_index=-1))
    else:
        loss_fct = torch.nn.DataParallel(CrossEntropyLoss(ignore_index=-1))

    output_train_file = os.path.join(args.output_dir, "train_logs.txt")
    if args.do_train:
        nb_total_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            ########### Evaluate at the end of each epoch
            if epoch % 2 == 0 and epoch > 0:
                model.eval()
                valid_loss, valid_accuracy = 0, 0
                for batch in tqdm(eval_dataloader, desc="Evaluating on validation set"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, lm_labels = batch
                    with torch.no_grad():
                        hidden_states = model(input_ids)[0]
                        logits = dense_layer(hidden_states)
                        # Shift so that tokens < n predict n
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = lm_labels[..., 1:].contiguous()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                        shift_labels.view(-1)).sum()
                    shift_logits = shift_logits.detach().cpu().numpy()
                    shift_labels = shift_labels.to('cpu').numpy()
                    num_predicitons = shift_labels.shape[0] * shift_labels.shape[1]
                    tmp_valid_accuracy = accuracy(shift_logits, shift_labels) / num_predicitons
                    valid_loss += loss.item()
                    valid_accuracy += tmp_valid_accuracy
                    # saving intermediate model
                    if args.do_save:
                        output_intermediate_model = os.path.join(args.output_dir, 'intermediate_model'+'_epoch_'+str(epoch)+'.bin')
                        torch.save(dense_layer, output_intermediate_model)
                print("")
                print("<<<<<<<<<<< Intermediate Evaluation >>>>>>>>>>>>>")
                print("Epoch : {} | validation loss : {} | validation accuracy : {} ".format(epoch,
                                                                                             valid_loss,
                                                                                             valid_accuracy))
            ######### End of intermediate evaluation

            ######### Start training
            import pdb
            tr_loss = 0
            nb_tr_steps = 0
            train_accuracy = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels = batch
                hidden_states = model(input_ids)
                pdb.set_trace()
                logits = dense_layer(hidden_states)
                if lm_labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = lm_labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)).sum()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                nb_tr_steps += 1
                nb_total_tr_steps += 1

                ######## Train Accuracy
                shift_logits = shift_logits.detach().cpu().numpy()
                shift_labels = shift_labels.to('cpu').numpy()
                num_predicitons = len([x for x in shift_labels.flatten() if x != -1])
                tmp_train_accuracy = accuracy(shift_logits, shift_labels)
                tmp_train_accuracy /= num_predicitons
                train_accuracy += tmp_train_accuracy
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, args.learning_rate)
                print("")
                # Save intermediate model for some steps
                if nb_total_tr_steps % args.save_step == 0 and args.do_save:
                    if nb_total_tr_steps == args.save_step:
                        tokenizer.save_vocabulary(log_dir_steps)

                    # Save a trained model
                    model_to_save = dense_layer.module if hasattr(dense_layer,
                                                                  'module') else dense_layer  # Only save the model it-self

                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file_steps = os.path.join(log_dir_steps, 'gpt2_intermediate_model_' + 'step_' + str(
                        nb_total_tr_steps) + '.bin')
                    torch.save(model_to_save, output_model_file_steps)
                # End
                # Completing logs files
                if step%args.save_train_logs == 0:
                    result = {'train_loss': exp_average_loss,
                              'step': step,
                              'epoch': epoch }
                    with open(output_train_file, "a") as writer:
                        for key in sorted(result.keys()):
                            writer.write("%s = %s, " % (key, str(result[key])))
                        writer.write("\n")
                # End
            # Completing logs files
            avg_train_accuracy = train_accuracy/nb_tr_steps
            avg_loss = tr_loss/nb_tr_steps
            result = {'avg train accuracy': avg_train_accuracy,
                      'avg loss': avg_loss,
                      'step': step,
                      'epoch': epoch}
            with open(output_train_file, "a") as writer:
                writer.write("End of Epoch |")
                for key in sorted(result.keys()):
                    writer.write(" %s = %s, " % (key, str(result[key])))
                writer.write("\n")
            # End
            print("Training accuracy {}".format(train_accuracy/nb_tr_steps))
            print(f"average loss for epoch {epoch}, {tr_loss/nb_tr_steps}")
            print("")

    # Save a trained model (only the (last) dense layer weights)
    if args.do_save:
        # Save a trained model, configuration and tokenizer
        model_to_save = dense_layer.module if hasattr(dense_layer,
                                                'module') else dense_layer  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, 'final_' + WEIGHTS_NAME )

        torch.save(model_to_save, output_model_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_eval:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, lm_labels = batch
            with torch.no_grad():
                hidden_states = model(input_ids)[0]
                logits = dense_layer(hidden_states)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)).sum()
            shift_logits = shift_logits.detach().cpu().numpy()
            shift_labels = shift_labels.to('cpu').numpy()
            num_predicitons = len([x for x in shift_labels.flatten() if x != -1])
            tmp_eval_accuracy = accuracy(shift_logits, shift_labels) / num_predicitons
            eval_loss += loss.item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        print("")
        print(f"At test time | eval_loss: {eval_loss} | eval_accuracy: {eval_accuracy}")
        train_loss = tr_loss / nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()