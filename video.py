import os
import warnings

import tensorflow as tf
import helper

from moviepy.editor import VideoFileClip


def run():
    image_shape = (160, 576)
    runs_dir = './runs'

    with tf.Session() as sess:
        #server = tf.train.Saver()
        meta_path = os.path.join(runs_dir, "model.ckpt.meta")
        checkpoint_path = os.path.join(runs_dir, "model.ckpt")
        meta_path = "/Users/yuishikawa/local/src/github/google-cloud-deep-learning-kit/carnd-project2/model.ckpt.meta"
        checkpoint_path = "/Users/yuishikawa/local/src/github/google-cloud-deep-learning-kit/carnd-project2/model.ckpt"
        saver = tf.train.import_meta_graph(meta_path)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        # print all tensors
        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        graph = tf.get_default_graph()
        logits = graph.get_tensor_by_name("logits:0")
        image_pl = graph.get_tensor_by_name("image_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        pipeline = helper.make_pipeline(sess, logits, keep_prob, image_pl, image_shape)

        clip = VideoFileClip('challenge.mp4')
        results = clip.fl_image(pipeline)
        results.write_videofile("./output.mp4")


if __name__ == '__main__':
    run()
