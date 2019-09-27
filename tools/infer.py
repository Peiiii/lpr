#!/usr/bin/env python3
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2

# from tfutils.helpers import load_module
from lpr.trainer import decode_beams


def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
  return graph

def display_license_plate(number, license_plate_img):
  size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
  text_width = size[0][0]
  text_height = size[0][1]

  height, width, _ = license_plate_img.shape
  license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0,
                                         0 if text_width < width else text_width - width,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
  cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

  return license_plate_img

def build_argparser():
  parser = ArgumentParser()
  parser.add_argument('--model', help='Path to frozen graph file with a trained model.', default='weights/graph.pb.frozen', required=False, type=str)
  parser.add_argument('--config', help='Path to a config.py',default='chinese_lp/config.py', required=False, type=str)
  parser.add_argument('--input_image','-i',default='data/demo/1.jpg', required=False,help='Image with license plate')
  # parser.add_argument('input_image',default='data/demo/1.jpg',help='Image with license plate')
  return parser



def main():


  # config = load_module(args.config)

  # image = cv2.imread(args.input_image)

  # graph = load_graph('weights/graph.pb.frozen')
  # config = load_module('chinese_lp/config.py')

  # dic=config.r_vocab
  # import json
  # with open('chinese_lp/charmap.json','r') as f:
  #   r_vocab=json.load(f)




  # img_file='data/demo/16.jpg'
  # image = cv2.imread('data/demo/4.jpg')

  args = build_argparser().parse_args()
  graph = load_graph(args.model)
  import json
  with open('chinese_lp/charmap.json', 'r') as f:
    r_vocab = json.load(f)
  from detect.predict import Yolo_test as Detector
  D=Detector()
  image=D.predict_from_file(args.input_image)

  if image is None:
    print('No plate detected.')
    return

  img = cv2.resize(image, (94, 24))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.float32(img)
  img = np.multiply(img, 1.0/255.0)

  input = graph.get_tensor_by_name("import/input:0")
  output = graph.get_tensor_by_name("import/d_predictions:0")

  with tf.Session(graph=graph) as sess:
    results = sess.run(output, feed_dict={input: [img]})
    # print(results)
    decoded_lp = decode_beams(results, r_vocab=r_vocab)[0]
    print(decoded_lp)

    # img_to_display = display_license_plate(decoded_lp, image)
    # cv2.imshow('License Plate', img_to_display)
    # cv2.waitKey(0)
  return decoded_lp

class Recongnizer:
  def __init__(self):
    self.graph=load_graph('weights/graph.pb.frozen')
    import json
    with open('chinese_lp/charmap.json', 'r') as f:
      self.r_vocab = json.load(f)
    self.sess=tf.Session(graph=self.graph)

  def predict_from_file(self,f):
    img=cv2.imread(f)
    return self.predict_from_file(img)
  def predict(self,img):
    graph=self.graph
    input = graph.get_tensor_by_name("import/input:0")
    output = graph.get_tensor_by_name("import/d_predictions:0")

    img = cv2.resize(img, (94, 24))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)
    img = np.multiply(img, 1.0 / 255.0)

    # with tf.Session(graph=graph) as sess:
    results = self.sess.run(output, feed_dict={input: [img]})
    decoded_lp = decode_beams(results, r_vocab=self.r_vocab)[0]
    return decoded_lp

if __name__ == "__main__":
  main()
