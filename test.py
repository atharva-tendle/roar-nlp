def test(args):
    
    test_loss = []
    test_batches = 0

    args.model.eval()
    for batch in args.dataloaders['test']:

        args.optim.zero_grad()

        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)

        outputs = args.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        test_loss.append(loss)
        test_batches += 1

    print("Testing loss: {}".format(sum(test_loss)/test_batches))
    
       
    

        
