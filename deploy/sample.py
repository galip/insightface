import face_model
import argparse
import cv2
import sys
import numpy as np

import mxnet as mx

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/galip/PycharmProjects/insightface/model-r50-am-lfw,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

def main():
  model = face_model.FaceModel(args)
  ctx = mx.cpu(0)
  if len(args.model) > 0:
      model =get_model(ctx, 112, args.model, 'fc1')

  #aligned face
  img = cv2.imread('/home/galip/PycharmProjects/insightface/deploy/Tom_Hanks_54745.png')
  # img = model.get_input(img)

  f1 = model.get_feature(img)
  print(f1[0:10])

  #aligned face
  img = cv2.imread('/home/galip/PycharmProjects/insightface/deploy/Tom_Hanks_54745.png')
  f2 = model.get_feature(img)
  print (f2)

if __name__ == '__main__':
    main()
