import os
import tensorflow as tf

from utils import TextLoader
from model import Seq2Seq

if __name__ == '__main__':
    text_loader = TextLoader('data')

    train_data = text_loader.load_data(128)

    # tf图reset，清除以前的图
    tf.reset_default_graph()

    # RNN层有多少个神经元，这个数目需要和embedding的维度一样，因为直接就过去了
    hidden_size = 300
    # 有多少个RNN层
    num_layers = 2

    # 使用我们自己创建的Seq2Seq模型
    model = Seq2Seq(hidden_size, num_layers, text_loader.embeddings)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 0
    n_epochs = 30
    save_dir = 'models\chat'

    saver = tf.train.Saver(tf.global_variables())
    model_file = tf.train.latest_checkpoint(save_dir)
    if model_file is not None:
        saver.restore(sess, model_file)

    while epoch < n_epochs:
        epoch += 1
        total_loss = 0 
        total_num_ins = 0
        pointer = 0
        for (encoder_inputs, encoder_length, mb_y, mb_y_mask) in train_data:
            pointer += 1
            decoder_inputs = mb_y[:, :-1]
            decoder_target = mb_y[:, 1:]
            loss = model.train(sess, encoder_inputs, encoder_length.sum(1), decoder_inputs, decoder_target, mb_y_mask[:, :-1])
            total_loss += loss
            total_num_ins += mb_y.shape[0]
            if (pointer % 100 == 0):
                print("batch {}, training loss: {}".format(pointer, total_loss / total_num_ins))
                checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
        print("training loss: {}".format(total_loss / total_num_ins))
