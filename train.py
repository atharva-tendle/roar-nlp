import time

def train_and_validate(args):
    print("Starting Training for {} epochs".format(epochs))

    train_loss, val_loss = [], []
    train_batches, val_batches = 0, 0

    for epoch in range(args.epochs):
        start_epoch = time.time()
        model.train()
        for batch in args.dataloaders['train']:
            args.optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = args.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            train_loss.append(loss)
            train_batches += 1
            loss.backward()
            args.optim.step()

        print("Epoch: {} - Training loss: {}".format(epoch, sum(train_loss)/train_batches))

        model.eval()
        for batch in args.dataloaders['val']:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = args.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            
            val_loss.append(loss)
            val_batches += 1

        print("Epoch: {} - Validation loss: {}".format(epoch, sum(val_loss)/val_batches))

        print("Time for Epoch: {}".format((time.time()-start_epoch)/60))