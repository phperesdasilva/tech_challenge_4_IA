from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

def train_model(model, optimizer, criterion, epochs, x_train, y_train):
    for epoch in range(epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)

        if epoch%1 == 0:
            print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(df, model, ticker, dataset, x_test, y_test):
    model.eval()

    y_test_pred = model(x_test)

    _, _, y_test_pred, y_test = dataset.invert_transform_data(y_test_pred=y_test_pred, y_test=y_test)

    # train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    # train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    return y_test_pred, y_test, test_rmse, test_mae, test_mape