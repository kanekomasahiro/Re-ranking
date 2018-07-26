import sys
sys.path.append('/home/masahirokaneko/S2V/src/')
import tensorflow as tf
import configuration
import encoder_manager
import json

import numpy as np


class Scoring():

    def __init__(self):
        self.FLAGS = tf.flags.FLAGS
        tf.flags.DEFINE_integer('gpu_id', 1, 'gpu_id')
        tf.flags.DEFINE_string('case', 'true', 'true or lower')
        tf.flags.DEFINE_string("eval_task", "MSRP",
                               "Name of the evaluation task to run. Available tasks: "
                               "MR, CR, SUBJ, MPQA, SICK, MSRP, TREC.")
        tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")
        tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
        tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
        tf.flags.DEFINE_boolean("use_norm", False,
                                "Normalize sentence embeddings during evaluation")
        tf.flags.DEFINE_integer("sequence_length", 50, "Max sentence length considered")
        tf.flags.DEFINE_string("model_config", "/home/masahirokaneko/S2V/model_hyperparameter.json", "Model configuration json")
        tf.flags.DEFINE_string("results_path", "/home/masahirokaneko/S2V/pre-trained_model/models/", "Model results path")
        tf.flags.DEFINE_string("Glove_path", "/home/masahirokaneko/pre-trained_model/dictionaries/GloVe", "Path to Glove dictionary")

        self.model = self.load_model(self.FLAGS)


    def load_model(self, FLAGS):
        tf.logging.set_verbosity(tf.logging.INFO)

        model = encoder_manager.EncoderManager()

        with open(FLAGS.model_config) as json_config_file:
            model_config = json.load(json_config_file)
        if type(model_config) is dict:
            model_config = [model_config]

        for mdl_cfg in model_config:
            model_config = configuration.model_config(mdl_cfg, mode="encode")
            model.load_model(model_config)

        return model


    def inference(self, pre_sentences, cur_sentences):
        pres = list()
        curs = list()

        for pre_sentence, cur_sentence in zip(pre_sentences, cur_sentences):
            if self.FLAGS.case == 'true':
                pres.append(pre_sentence)
                curs.append(cur_sentence)
            elif self.FLAGS.case == 'lower':
                pres.append(pre_sentence.lower())
                curs.append(cur_sentence.lower())

        _, pre_embs = self.model.encode(pres)
        cur_embs, _ = self.model.encode(curs)
        scores = np.sum((pre_embs*cur_embs), axis=1)
        return scores

        pre_embs1 = pre_embs[0][0]
        pre_embs2 = pre_embs[1][0]
        cur_embs1 = cur_embs[0][1]
        cur_embs2 = cur_embs[1][1]
        scores = [np.inner(np.mean(np.stack((pre_embs1[i], pre_embs2[i]), axis=1), axis=1),
                    np.mean(np.stack((cur_embs1[i], cur_embs2[i]), axis=1), axis=1)) for i in range(len(pre_embs1))]

        return scores
