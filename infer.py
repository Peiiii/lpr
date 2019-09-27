from tools.infer import main,Recongnizer
from detect.predict import Detector
import argparse


class Predictor:
	def __init__(self):
		self.detector=Detector()
		self.recongnizer=Recongnizer()
	def predict_from_file(self,f):
		img=self.detector.predict_from_file(f)
		if img is None:
			# print('No plate detected.')
			return
		y=self.recongnizer.predict(img)
		return y
	def predict_dir(self,dir):
		import os,glob
		fs=glob.glob(dir+'/*.jpg')
		fs.sort()
		ys=[]
		for i,f in enumerate(fs):
			y=self.predict_from_file(f)
			ys.append(y)
			print('%s, result: %s , file :%s'%(i,y,f))
		print('Finished.')
		return ys

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('--input_image','-i',default=None, required=False,help='Image with license plate')
	parser.add_argument('--input_dir','-d',default=None,required=False,help='Image directory contains license plate')
	args=parser.parse_args()
	P=Predictor()
	if args.input_image:
		y=P.predict_from_file(args.input_image)
		print('result: %s'%(y))
	elif args.input_dir:
		y=P.predict_dir(args.input_dir)
	else:
		print('No argument was given, recongnize from data/demo by default.')
		P.predict_dir('data/demo')