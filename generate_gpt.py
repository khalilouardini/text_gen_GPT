import argparse
import os
import torch
import logging
from tqdm import trange
import torch.nn.functional as F


from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(top_layer, model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    with torch.no_grad():
        for i in trange(length):
            hidden_states = model(prev)[0]
            logits = top_layer(hidden_states)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt2-medium',
                        choices=["openai-gpt", "gpt2", "gpt2-medium"], help='pretrained model name')
    parser.add_argument("--model_dir", type=str, help="path to model's local checkpoint",
                        default="/home/ouardinik/PycharmProjects/hyperlex/text_generation_GPT/logs/openai-gpt_2019-08-02_10:14:02.631911")
    parser.add_argument("--bin_filename", type=str, help="checkpoint's filename",
                        default="final_pytorch_model.bin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--run_parallel", action='store_true', help='whether to run on GPUs')

    args = parser.parse_args()

    assert os.path.exists(args.model_dir), "input model's path"

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    # Load Transformer
    logger.info("Load a trained model and vocabulary that you have fine-tuned")
    # BPE tokenizer and model
    if args.model_name == "openai-gpt":
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name)
        model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name)
    elif args.model_name == "gpt2" or args.model_name == "gpt2-medium":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # Device
    device = torch.device("cuda" if (torch.cuda.is_available() and args.run_parallel) else "cpu")
    n_gpu = torch.cuda.device_count()
    # Load top layer
    top_layer_path = os.path.join(args.model_dir, args.bin_filename)
    if device.type == "cpu":
        top_layer = torch.load(top_layer_path, map_location="cpu")
    else:
        top_layer = torch.load(top_layer_path)

    model.to(device), top_layer.to(device)
    model.eval(), top_layer.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    while True:
        context_tokens = []
        if not args.unconditional:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = tokenizer.encode(raw_text)
            generated = 0

            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    top_layer=top_layer,
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device
                )
                out = out[:, len(context_tokens):].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = tokenizer.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
        else:
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    top_layer=top_layer,
                    model=model, length=args.length,
                    context=None,
                    start_token=tokenizer.encoder['<|endoftext|>'],
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device
                )
                out = out[:, 1:].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = tokenizer.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    main()
