import os
import jieba
import tensorflow as tf

from utils import TextLoader
from model import Seq2Seq

if __name__ == '__main__':
    save_dir = 'models\chat' 
    text_loader = TextLoader('data')
    # RNN层有多少个神经元，这个数目需要和embedding的维度一样，因为直接就过去了
    hidden_size = 300
    # 有多少个RNN层
    num_layers = 2

    # 使用我们自己创建的Seq2Seq模型
    model = Seq2Seq(hidden_size, num_layers, text_loader.embeddings)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(save_dir)
        saver.restore(sess, model_file)

        talking = True
        while talking:
            question = input("> ")
            seg_list = jieba.cut(question, cut_all=False) 
            question = text_loader.get_words_id(seg_list)
            res = model.generate(sess, [question], [len(question)]).flatten()
            print(text_loader.get_words_by_id(res))
