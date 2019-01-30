import os
import io
import time
import jieba
import pickle

import numpy as np

class TextLoader():
    '''处理训练对话语料'''

    def __init__(self, data_dir, batch_size = 50, seq_length=25, words_num = 80000):
        '''初始化TextLoader
        data_dir: 存放训练数据的目录
        batch_size: 每个Batch的数据量
        seq_length: squence的长度
        words_num: 词语的总数量，最多30多万'''

        # TextLoader remembers its initialization arguments.
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        # 初始化一个空列表
        self.tensor_sizes = []

        # numpy的对话对照表的zip形式
        self.tensor_file_template = os.path.join(data_dir, "data_{}.npz")
        # 词汇表，我们从百度百科的词向量文件里创建
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        embedding_file = os.path.join(data_dir, "vocab.emb")
        # 每个训练数据集有多少数据
        sizes_file = os.path.join(data_dir, "sizes.pkl")

        # 获取所有.conv文件，并且计算conv文件的数量
        self.input_files = self._get_input_file_list(data_dir)
        self.input_file_count = len(self.input_files)

        if self.input_file_count < 1:
            raise ValueError("Input files not found. File names must end in '.conv'. Run utils.py to create conv file from raw data.")

        if self._preprocess_required(vocab_file, sizes_file, self.tensor_file_template, self.input_file_count):
            # If either the vocab file or the tensor file doesn't already exist, create them.
            t0 = time.time()
            print("Preprocessing the following files:")
            for i, filename in enumerate(self.input_files): print("   {}.\t{}".format(i+1, filename))

            # 创建词汇表
            print("Saving vocab file")
            self._save_vocab(vocab_file, embedding_file, words_num)

            for i, filename in enumerate(self.input_files):
                t1 = time.time()
                print("Preprocessing file {}/{} ({})... ".format(i+1, len(self.input_files), filename),
                        end='', flush=True)

                # 预处理文件，即处理词语到ID的映射
                self._preprocess(self.input_files[i], self.tensor_file_template.format(i))
                # 把每一个input的长度都追加到tensor_size里面
                self.tensor_sizes.append(self.tensor.size)

                print("done ({:.1f} seconds)".format(time.time() - t1), flush=True)


            # 记录每一个文件的词语数量，并用二进制的形式保存
            with open(sizes_file, 'wb') as f:
                print(self.tensor_sizes)
                pickle.dump(self.tensor_sizes, f)

            print("Processed input data: {:,d} characters loaded ({:.1f} seconds)".format(
                    self.tensor.size, time.time() - t0))
        else:
            # If the vocab file and sizes file already exist, load them.
            print("Loading vocab file...")
            self._load_vocab(vocab_file, embedding_file)
            print("Loading sizes file...")
            with open(sizes_file, 'rb') as f:
                self.tensor_sizes = pickle.load(f)

        # 计算每个训练数据文件可以分为多少个batch，以及batch的总数
        self.tensor_batch_counts = [n // (self.batch_size * self.seq_length) for n in self.tensor_sizes]
        self.total_batch_count = sum(self.tensor_batch_counts)
        print("Total batch count: {:,d}".format(self.total_batch_count))

        self.tensor_index = -1

    def _get_input_file_list(self, data_dir):
        '''得到数据列表'''
        suffixes = ['conv']
        input_file_list = []
        if os.path.isdir(data_dir):
            for walk_root, walk_dir, walk_files in os.walk(data_dir):
                for file_name in walk_files:
                    if file_name.startswith("."): continue
                    file_path = os.path.join(walk_root, file_name)
                    if file_path.endswith(suffixes[0]):
                        input_file_list.append(file_path)
        else: raise ValueError("Not a directory: {}".format(data_dir))
        return sorted(input_file_list)

    def _preprocess_required(self, vocab_file, sizes_file, tensor_file_template, input_file_count):
        '''判断目前的数据是否需要预处理'''
        if not os.path.exists(vocab_file):
            print("No vocab file found. Preprocessing...")
            return True
        if not os.path.exists(sizes_file):
            print("No sizes file found. Preprocessing...")
            return True
        for i in range(input_file_count):
            if not os.path.exists(tensor_file_template.format(i)):
                print("Couldn't find {}. Preprocessing...".format(tensor_file_template.format(i)))
                return True
        return False

    def _save_vocab(self, vocab_file, embedding_file, words_num):
        '''我们通过预训练的文件中找到前words_num个词'''
        vocabs = ['UNK', 'BOS', 'EOS']
        embeddings = [list(map(lambda x: round(x,6), np.random.normal(0, 1, 300).tolist())), [0] * 300, [0] * 300]
        vec_file = os.path.sep.join([self.data_dir, 'word2vec.txt'])
        with open(vec_file, 'rU', encoding='utf-8', errors='ignore') as source_file:
            for i in range(words_num):
                line = source_file.readline()
                if i > 0:
                    vector = line.split()
                    word = vector.pop(0)
                    # vector = [float(v) for v in vector]
                    # embeddings.append(vector)
                    vector_tmp = []
                    for v in vector:
                        try:
                            if (len(vector) != 300):
                                raise ValueError
                            vector_tmp.append(float(v))
                        except ValueError:
                            print("问题数据%d行%s.它的形状为%d" % (i,word,len(vector)))
                            break
                    else:
                        vocabs.append(word)
                        embeddings.append(vector_tmp)

        # 找不到的词后面用通配符替代
        self.chars = vocabs 
        self.vocab_size = len(vocabs)
        self.vocab = dict(zip(self.chars, range(self.vocab_size)))
        self.embeddings = embeddings

        # 将词汇表写到文件中
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)
        print("Saved vocab (vocab size: {:,d})".format(self.vocab_size))

        with open(embedding_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print("Saved embedding (embedding size: {:,d}x{:,d})".format(len(self.embeddings), 300))


    def _preprocess(self, input_file, tensor_file):
        '''对训练数据进行预处理
        把中文对话变成数字的形式,并且用numpy的ndarray形式存储，npz就是numpy的zip形式哦'''
        self.tensor = []
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as source_file:
            for line in source_file.readlines():
                words = [word.strip() for word in line.split()]
                ids = list(map(lambda x:self.vocab.get(x) if self.vocab.get(x) else 0, words))
                conversation = [1]
                conversation.extend(ids)
                conversation.append(2)
                self.tensor.append(conversation)

        self.tensor = np.array(self.tensor)
        # Compress and save the numpy tensor array to data.npz.
        np.savez_compressed(tensor_file, tensor_data=self.tensor)

        
    def _load_vocab(self, vocab_file, embedding_file):
        # Load the character tuple (vocab.pkl) to self.chars.
        # Remember that it is in descending order of character frequency in the data.
        print(vocab_file)
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        # Use the character tuple to regenerate vocab_size and the vocab dictionary.
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        # 加载词向量
        with open(embedding_file, 'rb') as f:
            self.embeddings = pickle.load(f)


    def load_conversations(self):
        '''得到数据列表'''
        suffixes = ['npz']
        dataset = []
        if os.path.isdir(self.data_dir):
            for walk_root, walk_dir, walk_files in os.walk(self.data_dir):
                for file_name in walk_files:
                    if file_name.startswith("."): continue
                    file_path = os.path.join(walk_root, file_name)
                    if file_path.endswith(suffixes[0]):
                        dataset.append(file_path)
        else: raise ValueError("Not a directory: {}".format(self.data_dir))

        train_questions = []
        train_answers = []
        for data_file in dataset:
            npz = np.load(data_file)
            train_questions.extend(npz['tensor_data'][:-1])
            train_answers.extend(npz['tensor_data'][1:])

        return train_questions, train_answers

    def load_data(self, batch_size):
        train_questions, train_answers = self.load_conversations()

        minibatches = self.get_minibatches(len(train_questions), batch_size)
        # display(minibatches)
        all_ex = []
        for minibatch in minibatches:
            q_sentences = [train_questions[t] for t in minibatch]
            a_sentences = [train_answers[t] for t in minibatch]
            mb_x, mb_x_mask = self.prepare_data(q_sentences)
            mb_y, mb_y_mask = self.prepare_data(a_sentences)
            all_ex.append((mb_x, mb_x_mask, mb_y, mb_y_mask))
            # display(all_ex)
        return all_ex

    def get_minibatches(self, n, minibatch_size, shuffle=False):
        '''把训练语料分成若干组，每组batch_size个数据
        第一组:1~batch_size,第二组:batch_size~2*batch_size'''
        idx_list = np.arange(0, n, minibatch_size)
        # display(idx_list)
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
        return minibatches

    def prepare_data(self, seqs):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        max_len = np.max(lengths)

        x = np.zeros((n_samples, max_len)).astype('int32')
        # 我们设置mask，为了最后算损失函数时候忽略mask掉的值
        x_mask = np.zeros((n_samples, max_len)).astype('float32')
        for idx, seq in enumerate(seqs):
            x[idx, :lengths[idx]] = seq
            x_mask[idx, :lengths[idx]] = 1.0
        return x, x_mask

    def get_words_id(self, words):
        ids = list(map(lambda x:self.vocab.get(x) if self.vocab.get(x) else 0, words))
        return ids

    def get_words_by_id(self, ids):
        my_dict = {v: k for k, v in self.vocab.items()}
        words = list(map(lambda x:my_dict.get(x) if my_dict.get(x) else '', ids))
        return words
   
class TextCreator():
    '''创建原始语料'''
    def __init__(self, data_dir):
        '''data_dir: 存放训练数据的目录'''
        self.data_dir = data_dir

    def create_dialog(self, func_name, file_name, suffix):
        '''创建标准的对话形式
        根据原始数据不同，调用不同的方法'''

        target_name = file_name + '.conv'
        file_name = file_name + '.' + suffix
        file_path = os.path.sep.join([self.data_dir, file_name])

        func_dict = {'process_mline': self.process_mline}
        processor = func_dict[func_name]
        dialogs = processor(file_path)

        target_path = os.path.sep.join([self.data_dir, target_name])
        with open(target_path, 'w', encoding='utf-8') as target_file:
            target_file.write(''.join(dialogs)) 

    def process_mline(self, file_path):
        dialogs = []
        print(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
            for line in source_file.readlines():
                if line.strip() == 'E':
                    # dialogs.append('\n')
                    continue
                elif line[0] == 'M':
                    dialog = line[2:].replace('/','').replace(',','，')
                    dialog_temp = []
                    for d in dialog.split('，'):
                        seg_list = jieba.cut(d, cut_all=False) 
                        seg_list = filter(lambda x: len(x.strip(' ')) > 0, seg_list)
                        dialog_temp.append(" ".join(seg_list))
                    dialogs.append(" ， ".join(dialog_temp))

        return dialogs

 

if __name__ == '__main__':
    text_creator = TextCreator('data')
    text_creator.create_dialog('process_mline', 'mline_dialog', 'txt')
