# Contract Generation
Finetuning OpenAI GPT and GPT2 on contract generation. We provide few samples (clauses extracted from contracts in english) in ```data/all_clauses.pkl``` that we use to finetune OpenAI GPT model pretrained on the WebText dataset. We build on the pytorch implementation of the model via the library pytorch-transformers.

```
git clone https://github.com/khalilouardini/text_gen_GPT/
```

# Install requirements
```
pip install -r requirements.txt
```

# Finetuning
```
bash finetune_gpt.sh
```
We use a parser (argparse) to input the parameters and configure the finetuning
Options:
   * --model_name, type=str, default='gpt2-medium' choices=['openai-gpt', 'gpt2', 'gpt2-medium'] : The model to finetune
                        
   * --do_train, action='store_true', default=True : Whether to run training
   * --do_eval, action='store_true', default=True: Whether to run evaluation on the evaluation set.
   * --do_save, action='store_true', default=True: Whether to save the trained model
   * --output_dir, type=str, default='logs': The output directory where the model predictions and checkpoints will be written.
   * --run_parallel", action="store_true", default=False: Whether to run on multiple GPU's"
   * --train_dataset, type=str, default='data/all_clauses.pkl': Training set 
   * --eval_dataset, type=str, default='data/all_clauses.pkl': Evaluation set
   * --seed, type=int, default=42 : Random Seed
   * --save_step, type=int, default=2 : Model saving frequency (the model will be saved if global_step%save_step ==0)
   * --save_train_logs, type=int, default=2 : Training logs saving frequency (the model will be saved if global_step%save_step ==0)
   * --num_train_epochs, type=int, default=20: Number of epochs
   * --train_batch_size, type=int, default=1 : Batch size during training
   * --eval_batch_size', type=int, default=5 : Batch size during evaluation
   * --learning_rate', type=float, default=6.25e-3 : Learning rate
   * --warmup_proportion', type=float, default=0.002 : proportion of the training using the warmup learning rate
   * --teacher_forcing', type=bool, default=True) : Whether to train the model with teacher forcing

# Generation
```
bash generate_gpt.sh
```
