import numpy as np
import random as python_random
import tensorflow as tf
import datetime
import argparse

np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to input data")
parser.add_argument("--log_dir", help="Directory to save logs")
parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints", default="./checkpoints")

class ConcatAttention(tf.keras.layers.Layer):
    def __init__(self, hid_dim=64, **kwargs):
        super(ConcatAttention, self).__init__(**kwargs)
        self.hid_dim = hid_dim

        self.w = tf.keras.layers.Dense(hid_dim)
        self.v = tf.keras.layers.Dense(1, activation="softmax")
            
    def call(self, hidden, outputs):
        hidden_repeat = tf.repeat(tf.expand_dims(hidden, axis=1), repeats=tf.shape(outputs)[1], axis=1)

        energy = tf.tanh(self.w(tf.concat((hidden_repeat, outputs), axis=2)))

        return self.v(energy)
    
    def get_config(self):
        config = super(ConcatAttention, self).get_config()
        
        config.update({
            "hid_dim": self.hid_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def audiornn_videornn(out_dim=8, input_shape1=(None, 128), input_shape2=(None, 128)):
    x_input1 = tf.keras.layers.Input(input_shape1, name="video")
    x_input2 = tf.keras.layers.Input(input_shape2, name="audio")

    # audio
    x_a1, _, _, _, _ = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=True, dropout=0.5))(x_input2)
    x_a1, h1, _, h2, _ = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=True, dropout=0.5))(x_a1)
    h = tf.concat((h1, h2), axis=1)
    w = ConcatAttention(256)(h, x_a1)

    x_a2 = tf.math.reduce_sum(w*x_a1, axis=1)
    x_a2 = tf.keras.layers.Dropout(0.5)(x_a2)

    # video
    x_v1, _, _, _, _ = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=True, dropout=0.5))(x_input1) # TODO: add masking
    x_v1 = tf.keras.layers.Dropout(0.5)(x_v1)

    x_v1, h, _ = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True, dropout=0.5)(x_v1)
    w = ConcatAttention(128)(h, x_v1)

    x_v2 = tf.math.reduce_sum(w*x_v1, axis=1)
    x_v2 = tf.keras.layers.Dropout(0.5)(x_v2)
 
    # concat video and audio features
    x = tf.concat((x_v2, x_a2), axis=1)
    x = tf.keras.layers.Dense(out_dim)(x)
    return tf.keras.Model([x_input1, x_input2], x)

def load_data(filename, actor_id):
    with open(filename, 'rb') as f:
        labels_map = {'angry':0, 'calm':1, 'disgust':2, 'fearful':3, 'happy':4, 'normal':1, 'sad':5, 'surprised':6}

        dataset = np.load(f, allow_pickle=True).T

        train_ids = dataset[2] < actor_id
        val_ids = dataset[2] >= actor_id

        train_data = {
            "video":np.array([video_feature for video_feature in dataset[3][train_ids]]),
            "audio":np.array([audio_feature.T for audio_feature in dataset[4][train_ids]]),
            "labels":np.array([labels_map[emotion] for emotion in dataset[1][train_ids]])
        }

        val_data = {
            "video":np.array([video_feature for video_feature in dataset[3][val_ids]]),
            "audio":np.array([audio_feature.T for audio_feature in dataset[4][val_ids]]),
            "labels":np.array([labels_map[emotion] for emotion in dataset[1][val_ids]])
        }

        return train_data, val_data

if __name__ == "__main__":
    args = parser.parse_args()

    if args.input is None:
        raise Exception("Input data not selected") 

    train_data, val_data = load_data(args.input, 19)

    emotion_model = audiornn_videornn(out_dim=7)

    opt = tf.keras.optimizers.Adam(0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    emotion_model.compile(opt, loss, metrics=["accuracy"])

    sheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr if (epoch+1) % 20 != 0 else lr*0.1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(args.checkpoint_dir, save_best_only=True)
    callbacks = [sheduler, checkpoint_callback]

    if not args.log_dir is None:
        log_dir = args.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)

    history = emotion_model.fit(x={"video":train_data["video"], "audio":train_data["audio"]}, 
            y=train_data["labels"], 
            epochs=60, 
            batch_size=128, 
            callbacks=callbacks,
            validation_data=((val_data["video"], val_data["audio"]), val_data["labels"]))

    best_epoch = np.argmin(history.history["val_loss"])
    train_acc = np.round(history.history["accuracy"][best_epoch], 2)
    val_acc = np.round(history.history["val_accuracy"][best_epoch], 2)

    print(f"Best weights: epoch {best_epoch}, train accuracy {train_acc}, val accuracy {val_acc}")