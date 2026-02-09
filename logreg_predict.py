if __name__ == "__main__":
    # get thhe dataset and the weights and maybe the output file "default: houses.csv"
    # need to know what scaler used (probubly the StandardScaler: mean , std)
    # transform the X_predict
    # load the weights
    # predict -> store in file "Index,Hogwarts House\n"

    f = open("houses.csv", "w+")
    f.write("Index,Hogwarts House\n")
    for i in range(0, len(y_pred)):
        f.write(f"{i},{y_pred[i]}\n")
