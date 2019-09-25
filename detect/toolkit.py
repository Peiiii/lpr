

def split_train_val_voc(data_dir,train_dir,val_dir,train_ratio=0.8,verbose=True):
	import glob,os,shutil,random
	imgs=glob.glob(data_dir+'/*.jpg')

	shutil.rmtree(train_dir) if os.path.exists(train_dir) else None
	shutil.rmtree(val_dir) if os.path.exists(val_dir) else None
	os.makedirs(train_dir)
	os.makedirs(val_dir)

	random.shuffle(imgs)
	num_train=int(len(imgs)*train_ratio)
	train_imgs=imgs[:num_train]
	val_imgs=imgs[num_train:]

	def copy_imgs_to(imgs,dst_dir):
		for i, img in enumerate(imgs):
			xml = img[:-4] + '.xml'
			img2 = dst_dir + '/' + os.path.basename(img)
			xml2 = dst_dir + '/' + os.path.basename(xml)
			shutil.copy(img, img2)
			shutil.copy(xml, xml2)

			if verbose:
				print('%s, %s' % (i, img))

	copy_imgs_to(train_imgs,train_dir)
	copy_imgs_to(val_imgs,val_dir)
	print('finished.')

def gen_demo_dir_from_classify_dataset(data_dir,output_dir,num=1000):
	import glob, os, shutil, random
	shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
	os.makedirs(output_dir)
	files=glob.glob(data_dir+'/*/*.jpg')
	random.shuffle(files)
	if not num==-1:
		files=files[:num]
	for i,f in enumerate(files):
		f2=output_dir+'/'+str(i)+'_'+os.path.basename(f)
		shutil.copy(f,f2)
		print('%s,%s'%(i,f))