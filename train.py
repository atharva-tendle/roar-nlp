import time

def train_and_validate(args, save_path="/work/cse896/atendle/model"):
    print("Starting Training for {} epochs".format(args.epochs))

    train_loss, val_loss = [], []
    train_batches, val_batches = 0, 0

    for epoch in range(args.epochs):
        start_epoch = time.time()
        args.model.train()
        for batch in args.dataloaders['train']:
            args.optim.zero_grad()
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            outputs = args.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            train_loss.append(loss)
            train_batches += 1
            loss.backward()
            args.optim.step()

        print("Epoch: {} - Training loss: {}".format(epoch, sum(train_loss)/train_batches))
    
    print("Time for Epoch: {}".format((time.time()-start_epoch)/60))

    args.model.save_pretrained(save_path)

    return args
        
