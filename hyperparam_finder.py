from loadmidi import load_midi_files_from
import midistuff
from modelmaker import Brain
import datetime

def find():
    find_lr_ep = True
    attempts = 12
    t = load_midi_files_from('./Chopin')

    t_seqs = []
    for seq in t:
        t_seqs.append(midistuff.get_sequences(seq))

    t_data = []
    for seq in t_seqs:
        t_data.append(midistuff.mus_seq_to_data(seq))

    if find_lr_ep:
        lr_ep_txt = open('./hyperparam_logs/lr_ep_finder_{}.txt'.format(datetime.datetime.now()), 'w')

    opts = ['rmsprop', 'adam']
    len_train_seq = 10
    num_lstm = 1
    num_dense = 0
    lstm_nodes = 2
    dense_nodes = 2
    lr, ep = 0.1, 0.1
    best_acc, best_loss = 0.0, 100.0

    best_num_lstm_layers_acc = 0
    best_num_dense_layers_acc = 0
    best_num_lstm_nodes_acc = 0
    best_num_dense_nodes_acc = 0
    best_training_seq_length_acc = 0

    best_num_lstm_layers_loss = 0
    best_num_dense_layers_loss = 0
    best_num_lstm_nodes_loss = 0
    best_num_dense_nodes_loss = 0
    best_training_seq_length_loss = 0

    best_lr_acc, best_ep_acc = 0.1, 0.1
    best_lr_loss, best_ep_loss = 0.1, 0.1
    test_num = 0
    for opty in opts:
        for _ in range(attempts):
            len_train_seq += 10
            for _ in range(attempts):
                lstm_nodes *= 2
                for _ in range(attempts):
                    dense_nodes *= 2
                    for _ in range(attempts):
                        num_dense += 1
                        for _ in range(attempts):
                            num_lstm += 1
                            for _ in range(attempts):
                                ep *= 0.1
                                for _ in range(attempts):
                                    lr *= 0.1
                                    test_num += 1
                                    print()
                                    print('--- Test {} ---'.format(test_num))
                                    if find_lr_ep:
                                        lr_ep_txt.write('\n--- Test {} ---\n'.format(test_num))
                                        lr_ep_txt.write('LSTM layers: {}, Dense layers: {}\n'.format(num_lstm, num_dense))
                                        lr_ep_txt.write('LSTM nodes: {}, Dense layers: {}\n'.format(lstm_nodes, dense_nodes))
                                        lr_ep_txt.write('Length of training sequence: {}\n'.format(len_train_seq))
                                        lr_ep_txt.write('Optimizer: {}\n'.format(opty))
                                        lr_ep_txt.write('Learning Rate: {}, Epsilon: {}\n'.format(lr,ep))
                                    rnn = Brain(t_data,
                                                gpu=True,
                                                train_seq_length=len_train_seq,
                                                num_lstm_layers=num_lstm,
                                                num_dense_layers=num_dense,
                                                lstm_nodes=lstm_nodes,
                                                dense_nodes=dense_nodes,
                                                opt=opty,
                                                learning_rate=lr,
                                                epsilon=ep)
                                    hist = rnn.train(num_of_epochs=10)
                                    # for acc in hist.history['acc']:
                                    #     if acc > best_acc:
                                    #         best_acc = acc
                                    #         best_lr_acc = lr
                                    #         best_ep_acc = ep
                                    #         best_num_lstm_layers_acc = num_lstm
                                    #         best_num_dense_layers_acc = num_dense
                                    #         best_num_lstm_nodes_acc = lstm_nodes
                                    #         best_num_dense_nodes_acc = dense_nodes
                                    #         best_training_seq_length_acc = len_train_seq
                                    for loss in hist.history['loss']:
                                        if loss < best_loss:
                                            best_loss = loss
                                            best_lr_loss = lr
                                            best_ep_loss = ep
                                            best_num_lstm_layers_loss = num_lstm
                                            best_num_dense_layers_loss = num_dense
                                            best_num_lstm_nodes_loss = lstm_nodes
                                            best_num_dense_nodes_loss = dense_nodes
                                            best_training_seq_length_loss = len_train_seq

    # print()
    # print('Best for accuracy:')
    # print('acc=',best_acc)
    # print('lstm layers={}, dense layers={}, lstm nodes={}, dense nodes={}'.format(best_num_lstm_layers_acc, best_num_dense_layers_acc, best_num_lstm_nodes_acc, best_num_dense_nodes_acc))
    # print('lr=',best_lr_acc, 'ep=',best_ep_acc)
    print()
    print('Best for loss:')
    print('loss=',best_loss)
    print('lstm layers={}, dense layers={}, lstm nodes={}, dense nodes={}'.format(best_num_lstm_layers_loss, best_num_dense_layers_loss, best_num_lstm_nodes_loss, best_num_dense_nodes_loss))
    print('lr=',best_lr_loss,'ep=',best_ep_loss)

    if find_lr_ep:
        # lr_ep_txt.write('\n\nBest for accuracy:')
        # lr_ep_txt.write('best acc={}\n'.format(best_acc))
        # lr_ep_txt.write('lstm layers={}, dense layers={}, lstm nodes={}, dense nodes={}\n'.format(best_num_lstm_layers_acc, best_num_dense_layers_acc, best_num_lstm_nodes_acc, best_num_dense_nodes_acc))
        # lr_ep_txt.write('lr=',best_lr_acc, 'ep=',best_ep_acc)
        lr_ep_txt.write('\nBest for loss:\n')
        lr_ep_txt.write('best loss={}\n'.format(best_loss))
        lr_ep_txt.write('lstm layers={}, dense layers={}, lstm nodes={}, dense nodes={}\n'.format(best_num_lstm_layers_loss, best_num_dense_layers_loss, best_num_lstm_nodes_loss, best_num_dense_nodes_loss))
        lr_ep_txt.write('lr={}, ep={}\n'.format(best_lr_loss, best_ep_loss))
        lr_ep_txt.close()

if __name__ == "__main__": find()
