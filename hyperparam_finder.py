from loadmidi import load_midi_files_from
import midistuff
from modelmaker import Brain
import datetime

def find():
    find_lr_ep = True
    t = load_midi_files_from('./Test_Midi')

    t_seqs = []
    for seq in t:
        t_seqs.append(midistuff.get_sequences(seq))

    t_data = []
    for seq in t_seqs:
        t_data.append(midistuff.mus_seq_to_data(seq))

    if find_lr_ep:
        d = datetime.datetime.now()
        lr_ep_txt = open('./hyperparam_logs/lr_ep_finder_{}.txt'.format(str(d.date()) +
                                                                        '_' +
                                                                        str(d.time()).split('.')[0]),
                                                                        'w')

    opts = ['rmsprop', 'adam']
    len_train_seq = [10, 20, 30, 40, 50]
    lstm_layers = [4, 5, 6, 7]
    dense_layers = [3, 4, 5, 6]
    num_lstm_nodes = [128, 256, 512, 1028]
    num_dense_nodes = [128, 256, 512, 1024]
    lrs = [0.005]#, 5e-5, 1e-5, 5e-7, 1e-7, 5e-9, 1e-9, 5e-11, 1e-11]
    eps = [0.5]#, 5e-5, 1e-5, 5e-7, 1e-7, 5e-9, 1e-9, 5e-11, 1e-11]
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
        for train_seq_length in len_train_seq:
            for lstm_nodes in num_lstm_nodes:
                for dense_nodes in num_dense_nodes:
                    for num_dense in dense_layers:
                        for num_lstm in lstm_layers:
                            for ep in eps:
                                for lr in lrs:
                                    test_num += 1
                                    print()
                                    print('--- Test {} ---'.format(test_num))
                                    print('train seq len={}, lstm nodes={}, dense nodes={}, dense layers={}, lstm layers={}, ep={}, lr={}'.format(train_seq_length,lstm_nodes,dense_nodes,num_dense,num_lstm,ep,lr))
                                    if find_lr_ep:
                                        lr_ep_txt.write('\n--- Test {} ---\n'.format(test_num))
                                        lr_ep_txt.write('LSTM layers: {}, Dense layers: {}\n'.format(num_lstm, num_dense))
                                        lr_ep_txt.write('LSTM nodes: {}, Dense layers: {}\n'.format(lstm_nodes, dense_nodes))
                                        lr_ep_txt.write('Length of training sequence: {}\n'.format(train_seq_length))
                                        lr_ep_txt.write('Optimizer: {}\n'.format(opty))
                                        lr_ep_txt.write('Learning Rate: {}, Epsilon: {}\n'.format(lr,ep))
                                    rnn = Brain(t_data,
                                                gpu=False,
                                                train_seq_length=train_seq_length,
                                                num_lstm_layers=num_lstm,
                                                num_dense_layers=num_dense,
                                                lstm_nodes=lstm_nodes,
                                                dense_nodes=dense_nodes,
                                                opt=opty,
                                                learning_rate=lr,
                                                epsilon=ep)
                                    hist = rnn.train(num_of_epochs=50)
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
                                            best_training_seq_length_loss = train_seq_length

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
