def train(model, optimizer, criterion, train_loader, num_epochs, device):
    model.train()
    running_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d/5] loss: %.3f' %
              (epoch + 1, running_loss / 2000))
        running_loss = 0.0

    print('Finished Training')
