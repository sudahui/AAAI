import tensorflow as tf
import random
import pickle
import numpy as np

"""
    labels: vocab index labels, max_length*batch_size, padding labels are 0s
    length: length of every sequence, batch_size
"""


class data_producer:
    def __init__(self, frs, num_epochs):
        self.__dict__.update(locals())
        self.file_queue = tf.train.string_input_producer(frs)  # , num_epochs=self.num_epochs)
        self.reader = tf.TFRecordReader()

    # label: vocabulary index list, len
    # convert one-line dialogue into a tf.SequenceExample
    def __make_example(self, label, r_label, length):
        ex = tf.train.SequenceExample()
        # one sequence
        ex.context.feature["len"].int64_list.value.append(length)
        for w in label:
            ex.feature_lists.feature_list['seq'].feature.add().int64_list.value.append(w + 1)  # prevent 0 for padding
        for r in r_label:
            ex.feature_lists.feature_list['label'].feature.add().int64_list.value.append(r)
        return ex

    """
    slice a dialogue with size limit and make example
    slide over the whole dialogue, the next sequence share the end of the last sequence
    all results are padded with 0s if the length is less than limit
    """

    def slice_dialogue(self, dialogue, r, limit):
        exs = []
        start = 0
        while ((len(dialogue) - 1) > start):
            length = limit
            if start + limit > len(dialogue):  # padding 0
                length = len(dialogue) - start
                dialogue.extend([-1] * (start + limit - len(dialogue)))
                r.extend([0] * (start + limit - len(r)))
            ex = self.__make_example(dialogue[start:start + limit], r[start:start + limit], length)
            start += (limit - 1)
            exs.append(ex)
        return exs

    # labels: list of label(length), save as tfrecord form
    def save_record(self, labels, r_labels, fout, limit=80):
        writer = tf.python_io.TFRecordWriter(fout)
        num_record = 0
        for dialogue, r in zip(labels, r_labels):
            if num_record % 100 == 0:
                print(num_record)
            for ex in self.slice_dialogue(dialogue,r, limit):
                num_record += 1
                writer.write(ex.SerializeToString())
        writer.close()
        print('num_record:', num_record)

    # read from a list of TF_Record files frs, return a parsed Sequence_example
    # Every Sequence_example contains one dialogue
    # emotion=0 indicates no emotion information stored
    def __read_record(self, emotion=0):
        # first construct a queue containing a list of filenames.
        # All data can be split up in multiple files to keep size down
        # serialized_example is a Tensor of type string.
        _, serialized_example = self.reader.read(self.file_queue)
        # create mapping to parse a sequence_example
        context_features = {'len': tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_features = {'seq': tf.FixedLenSequenceFeature([], dtype=tf.int64),'label':tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        # sequences is a sequence_example for one dialogue
        length, sequences = tf.parse_single_sequence_example(
            serialized_example,
            context_features=context_features,
            sequence_features=sequence_features)
        return length, sequences

    """
    get the next batch from a list of files frs
    emotion=0 indicates no emotion information stored
    return length, labels
    """
    def batch_data(self, batch_size, emotion=0):
        length, sequences = self.__read_record(emotion)  # one-line dialogue
        print sequences
        # shuffled batch
        batched_seq,batched_label, batched_len = tf.train.shuffle_batch(
            tensors=[sequences['seq'], sequences['label'], length['len']],
            batch_size=batch_size, capacity=50000, min_after_dequeue=10000, shapes=[[100],[100],[]]
        )

        return tf.transpose(batched_seq, perm=[1, 0]),tf.transpose(batched_label, perm=[1, 0]), batched_len

if __name__ == '__main__':
    producer = data_producer(['./tfrecord/input.tfrecord'], 1)
    with open('/home/suhui/project/Ubuntudata/Training.dialogues.pkl', 'rb') as f:
        with open('label.dialogues.pkl', 'rb') as f1:
            
            labels = pickle.load(f)
            r_labels = pickle.load(f1)
            for i in range(30):
                print(len(labels[i]))
                print(len(r_labels[i]))
            random.seed(1)
            random.shuffle(labels)
            random.seed(1)
            random.shuffle(r_labels)
            producer.save_record(labels,r_labels, 'tfrecord_revised_labels')
