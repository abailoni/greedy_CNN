import matplotlib.pyplot as plt

def plot_loss(net, name_file="loss.pdf"):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    ax.plot(train_loss, label='train loss')
    ax.plot(valid_loss, label='valid loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='best')
    fig.savefig(name_file)
