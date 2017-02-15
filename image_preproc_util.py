import matplotlib.pyplot as plt
import numpy as np
import os
import time
import PIL
from PIL import Image
import scipy.ndimage
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
import pandas as pd
import scipy.ndimage
import shutil
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc
import pickle

FBRESNET = 1000
NAGADOMI = 2000


class data_batch_gen:

	def __init__(self,batch_size=100,path_dir=None,path_file=None):
		self.reader_pos = 0
		self.batch_size = batch_size
		self.path_dir = path_dir
		self.path_file = path_file

		fp = open(path_file)
		F = fp.readlines()
		fp.close()
		del F[0]

		self.files_labels = F
		
		# for j,i in enumerate(F):
		# 	g = i.strip().split(',')
		# 	filename = g[0]+'.jpg'
		# 	self.files_labels.append((filename,int(g[1]))

	def next_batch(self,pos=None,batch_size=None):
		if pos is not None:
			self.reader_pos = pos
		if batch_size is not None:
			self.batch_size = batch_size
		if self.reader_pos==len(self.files_labels):
			return None,self.reader_pos


		ids = []
		D = []
		label = []

		cur_pos = self.reader_pos
		new_pos = min(len(self.files_labels),self.reader_pos+self.batch_size)

		for e in self.files_labels[cur_pos:new_pos]:
			e = e.strip().split(',')
			label.append(int(e[1]))
			img = np.array(plt.imread(os.path.join(self.path_dir,e[0]+'.jpg')),dtype='uint8')
			D.append(img)
			ids.append(e[0])
		D = np.asarray(D)
		self.reader_pos = new_pos
		return D,ids


def viz_images(ip_dir):
	Y = os.listdir(ip_dir)
	for j,i  in enumerate(Y):
		PIL.Image.open(os.path.join(ip_dir,i)).show()
		time.sleep(1)
		

def get_data(images_dir,labelfile_path,one_hot=True,no=1000,load=False,batch_size=None,file_paths=None):
	if load:
		D = np.load(file_paths[0])
		L = np.load(file_paths[1])
		if len(L.shape)==1:
			enc = OneHotEncoder()
			label = np.expand_dims(L,1)
			enc.fit(label)
			L = np.array(enc.transform(label).toarray(),dtype='uint8')
		return D,L
	fp = open(labelfile_path)
	F = fp.readlines()
	fp.close()
	del F[0]
	#F = F[1:1000]

	label = []
	D = []
	for j,i in enumerate(F):
		g = i.strip().split(',')
		filename = g[0]+'.jpg'
		if(j%50==0):
			print(j)
		#assert unicode(g[0], 'utf-8').isnumeric()

		label.append(int(g[1]))
		
		img = scipy.ndimage.imread(os.path.join(images_dir,filename))
		D.append(img)

	D = np.asarray(D)
	print ('Data dimensions',D.shape)
	print ('Num of classes:',len(set(label)))

	if one_hot:
		enc = OneHotEncoder()
		label = np.expand_dims(label,1)
		enc.fit(label)
		label = np.array(enc.transform(label).toarray(),dtype='uint8')
	return D,label

def demean(data):
	mean = np.mean(data,axis=0,keepdims=1)
	return data-mean
def standardize(data,mean=None,sigma=None,apply=None):
	if not apply:
		mean = np.mean(data,axis=0,keepdims=1)
		sigma = np.std(data,axis=0,keepdims=1)
		return np.divide(data-mean,sigma),mean,sigma
	return np.divide(data-mean,sigma),None,None
def put_images(ip_dir,op_dir,path_file):
	df = pd.read_csv(path_file)
	for index, row in df.iterrows():
		if index%1000==0:
			print(index)
		id_ = str(row['Id'])
		file_ = os.path.join(ip_dir,id_)
		if '.jpg' not in file_:
			file_= file_+'.jpg'
		shutil.copy(file_,op_dir)	
def copy_files(ip_dir,op_dir,path_file,new_foldernames=['c1','c2','c3','c4'],dataframe=None,type=FBRESNET):
	if type==FBRESNET:
		for i in new_foldernames:
			if not os.path.exists(os.path.join(op_dir,i)):
				os.makedirs(os.path.join(op_dir,i))
	if dataframe is not None:
		df = dataframe
	else:
		df = pd.read_csv(path_file)
	for index, row in df.iterrows():
		if index%1000==0:
			print (index)
		l = int(row['label'])
		id_ = str(row['Id'])
		file_ = os.path.join(ip_dir,id_)
		if '.jpg' not in file_:
			file_= file_+'.jpg'
		if type==FBRESNET:
			folder_name = new_foldernames[l-1]
			destination = os.path.join(op_dir,folder_name)
		else:
			destination = op_dir

		shutil.copy(file_,destination)
def mytrain_test_split(df,test_size):
	np.random.seed(123)
	ind = range(len(df.index)/4)
	np.random.shuffle(ind)
	rand_ind = np.multiply(ind,4)
	train_ind = rand_ind[1:int((1-test_size)*len(rand_ind))]
	test_ind = rand_ind[int((1-test_size)*len(rand_ind)):]

	train_ind = np.array(np.hstack([train_ind,np.add(train_ind,1),np.add(train_ind,2),np.add(train_ind,3)]),dtype='int')
	test_ind = np.array(np.hstack([test_ind,np.add(test_ind,1),np.add(test_ind,2),np.add(test_ind,3)]),dtype='int')

	return df.ix[train_ind],df.ix[test_ind]

def split_train_val_folders_fbresnet(ip_dir,dataset_root_dir,path_file,new_foldernames=['c1','c2','c3','c4']):
	# df = pd.read_csv(path_file)
	# train, val = mytrain_test_split(df, test_size = 0.2)
	# train_folderpath = os.path.join(dataset_root_dir,'train')
	# val_folderpath = os.path.join(dataset_root_dir,'val')
	# if not os.path.exists(train_folderpath):
	# 	os.makedirs(train_folderpath)
	# 	for i in new_foldernames:
	# 		os.makedirs(os.path.join(train_folderpath,str(i)))
	# if not os.path.exists(val_folderpath):
	# 	os.makedirs(val_folderpath)
	# 	for i in new_foldernames:
	# 		os.makedirs(os.path.join(val_folderpath,str(i)))
	# copy_files(ip_dir,train_folderpath,None,new_foldernames,train)
	# copy_files(ip_dir,val_folderpath,None,new_foldernames,val)
	split_train_val_folders_nagadomi(ip_dir,dataset_root_dir,path_file)
	print('Splitting train-val followed by uniform sampling as per nagadomi done for fbresnet!')
	print('Now we will split train_uni into train/<labels from 1 to 4> and similarly for val_uni....')
	fb_train_folderpath = os.path.join(dataset_root_dir,'train')
	fb_val_folderpath = os.path.join(dataset_root_dir,'val')

	assert not os.path.exists(fb_train_folderpath)
	os.makedirs(fb_train_folderpath)
	assert not os.path.exists(fb_val_folderpath)
	os.makedirs(fb_val_folderpath)

	copy_files(os.path.join(dataset_root_dir,'train_uni'),fb_train_folderpath,\
		os.path.join(dataset_root_dir,'train_uni_labels.csv'),new_foldernames,dataframe=None,type=FBRESNET)
	print('Fbresnet type train data is ready at supplied root directory : %s',dataset_root_dir)
	copy_files(os.path.join(dataset_root_dir,'val_uni'),fb_val_folderpath,\
		os.path.join(dataset_root_dir,'val_uni_labels.csv'),new_foldernames,dataframe=None,type=FBRESNET)
	print('Fbresnet type val data is ready at supplied root directory : %s',dataset_root_dir)
	print('Now we will delete unwanted directories and files....')
    # shutil.rmtree(os.path.join(dataset_root_dir,'train_uni'))
    # print('train_uni/ deleted')
    # shutil.rmtree(os.path.join(dataset_root_dir,'val_uni'))
    # print('val_uni/ deleted')
    # shutil.rmtree(os.path.join(dataset_root_dir,'train_pre'))
    # print('train_pre/ deleted')
    # shutil.rmtree(os.path.join(dataset_root_dir,'val_pre'))
    # print('val_pre/ deleted')
    # os.remove(os.path.join(dataset_root_dir,'train_uni_labels.csv'))
    # os.remove(os.path.join(dataset_root_dir,'val_uni_labels.csv'))
    # os.remove(os.path.join(dataset_root_dir,'train_pre_labels.csv'))
    # os.remove(os.path.join(dataset_root_dir,'val_pre_labels.csv'))
    # print('All csv files deleted..Now we are good to go for fbresnet.Here is the root directory for fbresnet command line interface : %s',\
    # 	dataset_root_dir)

def split_train_val_folders_nagadomi(ip_dir,dataset_root_dir,path_file):
	df = pd.read_csv(path_file)
	train, val = mytrain_test_split(df, test_size = 0.1)
	train_folderpath = os.path.join(dataset_root_dir,'train_pre')
	val_folderpath = os.path.join(dataset_root_dir,'val_pre')
	train_folderpath_uni = os.path.join(dataset_root_dir,'train_uni')
	val_folderpath_uni = os.path.join(dataset_root_dir,'val_uni')

	if not os.path.exists(train_folderpath):
		os.makedirs(train_folderpath)
	if not os.path.exists(val_folderpath):
		os.makedirs(val_folderpath)
	train.to_csv(os.path.join(dataset_root_dir,'train_pre_labels.csv'),index=False)
	val.to_csv(os.path.join(dataset_root_dir,'val_pre_labels.csv'),index=False)
	copy_files(ip_dir,train_folderpath,None,None,train,NAGADOMI)
	copy_files(ip_dir,val_folderpath,None,None,val,NAGADOMI)

	print('Train-Val split done....now we will uniform sample both...')
	if not os.path.exists(train_folderpath_uni):
		os.makedirs(train_folderpath_uni)
	if not os.path.exists(val_folderpath_uni):
		os.makedirs(val_folderpath_uni)

	uniform_sample(train_folderpath,os.path.join(dataset_root_dir,'train_pre_labels.csv'),\
		os.path.join(dataset_root_dir,'train_uni_labels.csv'),train_folderpath_uni)
	print('Uniform sampling done on train!')
	uniform_sample(val_folderpath,os.path.join(dataset_root_dir,'val_pre_labels.csv'),\
		os.path.join(dataset_root_dir,'val_uni_labels.csv'),val_folderpath_uni)
	print('Uniform sampling done on val!')

	

def rescale_and_patch(ip_dir,op_dir,new_size):
	Y = os.listdir(ip_dir)
	for j,i  in enumerate(Y):
		if(j%500==0):
			print(j)
		img = Image.open(os.path.join(ip_dir,i))
		w,h = img.size
		if w>h:
			scale = new_size/float(h)
			img = img.resize((int(scale*w),new_size), PIL.Image.ANTIALIAS)
			w,h = img.size
			res = int((w-new_size)/2)
			img = img.crop((res,0,res+new_size,h))
		else:
			scale = new_size/float(w)
			img = img.resize((new_size,int(scale*h)), PIL.Image.ANTIALIAS)
			w,h = img.size
			res = int((h-new_size)/2)
			img = img.crop((0,res,w,res+new_size))
		img.save(os.path.join(op_dir,i))
def uniform_sample(ip_dir,labelfile_path,new_lablefilepath=None,op_dir=None,luatype=False):
	df = pd.read_csv(labelfile_path)
	arr = [[],[],[],[]]
	#2d array 
	for index,row in df.iterrows():
		arr[int(row['label'])-1].append((str(row['Id']),int(row['label'])))

	l1=len(arr[0])
	l2=len(arr[1])
	l3=len(arr[2])
	l4=len(arr[3])

	L=max(l1,l2,l3,l4)
	print (L)
	Arr = []
	np.random.shuffle(arr[0])
	np.random.shuffle(arr[1])
	np.random.shuffle(arr[2])
	np.random.shuffle(arr[3])

	for x in range(0, L): 
		for j in range(0,4):
			vu = x
			if x>=len(arr[j]):
				vu = x%len(arr[j]) 
			Arr.append(arr[j][vu])

	if op_dir is None:	
		label = []
		D = []
		for i in range(4*L):
			id_ = Arr[i][0]
			if(i%500==0):
				print (i)
			filepath_ = os.path.join(ip_dir,id_+'.jpg')
			img = scipy.ndimage.imread(filepath_)
			if luatype:
				D.append(np.moveaxis(img,2,0))
			else:
				D.append(img)
			label.append(Arr[i][1])

		D = np.asarray(D)
		label=np.asarray(label)

		rand_ind = list(np.random.permutation(D.shape[0]))
		D = D[rand_ind]
		label = label[rand_ind]

		return D,label
	else:
		if new_lablefilepath is None:
			print('Error,new_lablefilepath needs to be given as argument')
			return 0
		Id = []
		labels = []
		uniq_id = {}
		for i in range(4*L):
			id_ = Arr[i][0]
			l_ = Arr[i][1]
			if id_ not in uniq_id:
				uniq_id[id_] = 0
			else:
				uniq_id[id_] = uniq_id[id_] + 1
			
			if(i%500==0):
				print (i)

			filepath_ = os.path.join(ip_dir,id_+'.jpg')
			shutil.copy(filepath_,op_dir)
			dst_filepath_ = os.path.join(op_dir,id_+'.jpg')
			new_dst_filepath_ = os.path.join(op_dir,id_+'_'+str(uniq_id[id_])+'.jpg')
			os.rename(dst_filepath_,new_dst_filepath_)

			Id.append(id_+'_'+str(uniq_id[id_]))
			labels.append(l_)
		df = pd.DataFrame({'Id':Id,'label':labels})
		df.to_csv(new_lablefilepath,index=False)
		print('Size of uniform Sampled data : %s',df.shape[0])
def augment_by_rotate(images_dir,labelfile_path,op_dir,new_labelfilepath):
	name = []
	label = []
	df = pd.read_csv(labelfile_path)
	for index,g in df.iterrows():
		filename = str(g[0])+'.jpg'
		if(index%500==0):
			print (index)

		src_im = Image.open(os.path.join(images_dir,filename))

		filename = str(g[0])+'.jpg'
		name.append(filename[:-4])
		label.append(int(g[1]))
		src_im.save(os.path.join(op_dir,filename))

		filename = str(g[0])+'r.jpg'
		name.append(filename[:-4])
		src_im.rotate(-90,resample=Image.BILINEAR).save(os.path.join(op_dir,filename))
		
		filename = str(g[0])+'l.jpg'
		name.append(filename[:-4])
		src_im.rotate(90,resample=Image.BILINEAR).save(os.path.join(op_dir,filename))

		if int(g[1])==1:
			label.append(2)
			label.append(2)
		elif int(g[1])==2:
			label.append(1)
			label.append(1)
		else:
			label.append(int(g[1]))			
			label.append(int(g[1]))			
		
		filename = str(g[0])+'an.jpg'
		name.append(filename[:-4])
		label.append(int(g[1]))			
		src_im.rotate(180,resample=Image.BILINEAR).save(os.path.join(op_dir,filename))

	print 'Augmentation Successful'

	df = pd.DataFrame({'Id':name,'label':label})
	df.to_csv(new_labelfilepath, index=False)
def summary_statistics(ip_dir):
	h = []
	w = []
	c = []
	Y = os.listdir(ip_dir)
	print (len(Y))
	for j,i  in enumerate(Y):
		if(j%500==0):
			print(j)
		img = plt.imread(os.path.join(ip_dir,i))
		h.append(img.shape[0])
		w.append(img.shape[1])
		c.append(img.shape[2])
	return h,w,c
def image_statistics(ip_dir):
	Y = os.listdir(ip_dir)
	# ind = range(len(Y))
	# np.random.shuffle(ind)
	# sum_rgb = np.zeros((3,),dtype='float')
	# N = 0
	# A =[]
	# for j,i  in enumerate(Y):
	# 	if(j%500==0):
	# 		print(j)
	# 	img = np.divide(scipy.ndimage.imread(os.path.join(ip_dir,i)),255.0)
	# 	sum_rgb = sum_rgb + np.sum(img,axis=(0,1))
	# 	N = N + img.shape[0]*img.shape[1]
	# 	A.append(np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2])))
	# mean_rgb = sum_rgb/float(N)
	# print('Loop for mean over!Mean is ',mean_rgb)
	
	# sumvar = np.zeros((3,),dtype='float')
	# for j,i  in enumerate(Y):
	# 	if(j%500==0):
	# 		print(j)
	# 	img = np.divide(scipy.ndimage.imread(os.path.join(ip_dir,i)),255.0)
	# 	sumvar = sumvar + (np.sum(img,axis=(0,1)-mean_rgb)**2
	# std_rgb = sumvar/float(N)
	A=[]
	for j,i  in enumerate(Y):
		if(j%500==0):
			print(j)
		img = np.divide(scipy.ndimage.imread(os.path.join(ip_dir,i)),255.0)
		A.append(np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2])))
	A = np.vstack(A)
	print('A.shape',A.shape)
	mean_rgb = np.mean(A,axis=(0))
	std_rgb = np.std(A,axis=(0))
	print('mean:',mean_rgb,std_rgb)
	pca = PCA()
	pca.fit(A)
	print('PCA done')
	print pca.n_components
	print pca.explained_variance_ratio_
# def PCA_comps(data):
# 	data = np.reshape(data,(data.shape[0]*data.shape[1]*data.shape[2],data.shape[3]))
# 	print('Data reshaped!')
# 	pca = PCA()
# 	pca.fit(data)
# 	print('PCA done')
# 	print pca.n_components
# 	print pca.explained_variance_ratio_

def ten_crop(images_dir,crop_aug_dir,labelfile_path,new_labelfilepath,return_nparray=False): 
	if not os.path.exists(crop_aug_dir):
		os.makedirs(crop_aug_dir)
	fp = open(labelfile_path)
	F = fp.readlines()
	fp.close()
	del F[0]
	label = []
	name = []
	imgdata = []
	for j,i in enumerate(F):
		g = i.strip().split(',')
		filename = g[0]+'.jpg'
		if(j%100==0):
			print j
		
#		im = scipy.ndimage.imread(os.path.join(images_dir,filename),mode='RGB')
		src_im = Image.open(os.path.join(images_dir,filename))
		width,height = src_im.size

		for k in range(10):

			if (height>width):
				dst_im = Image.new("RGB", (32,32))
				im = src_im.convert('RGB')	
				res = im.resize((height,32),Image.ANTIALIAS)
				add = (height-32)/9
				res = im.crop((0,(k*add),32,(k*add)+32))
				dst_im = res
				name.append(g[0]+'#'+str(k))
				label.append(int(g[1]))
				if return_nparray:
					imgdata.append(np.moveaxis(np.array(dst_im),2,0))
				else:
					dst_im.save(os.path.join(crop_aug_dir,g[0]+'#'+str(k)+'.jpg'))
#				imsave(os.path.join(crop_aug_dir,(g[0]+'#'+str(k)+'.jpg')),res)
			else:
				dst_im = Image.new("RGB", (32,32))
				im = src_im.convert('RGB')	
				res = im.resize((32,width),Image.ANTIALIAS)
				add = (width-32)/9
				res = im.crop(((k*add),0,(k*add)+32,32))
				dst_im = res
				name.append(g[0]+'#'+str(k))
				label.append(int(g[1]))
				if return_nparray:
					imgdata.append(np.moveaxis(np.array(dst_im),2,0))
				else:
					dst_im.save(os.path.join(crop_aug_dir,g[0]+'#'+str(k)+'.jpg'))
				
#				imsave(os.path.join(crop_aug_dir,(g[0]+'#'+str(k)+'.jpg')),res)

	print 'Ten Crop Successful'
	if return_nparray:
		return np.asarray(imgdata),np.asarray(label,dtype = 'uint8')
	else:
		df = pd.DataFrame({'Id':name,'label':label})
		df.to_csv(new_labelfilepath, index=False)
def zca_images(X,y):
	# X = np.moveaxis(X,1,-1)
	# #datagen = ImageDataGenerator(vertical_flip=True,zca_whitening = True,shear_range=0.1,zoom_range=0.2,horizontal_flip=True)
	# datagen = ImageDataGenerator(zca_whitening = True,zoom_range=0.2)
	# # test_datagen = ImageDataGenerator(rescale=1./255)
	# datagen.fit(X)
	# print('Data fit')
	# # fits the model on batches with real-time data augmentation:
	# #model.fit_generator(datagen.flow(X_,y, batch_size=100,save_to_dir='./wresnet/keras_aug_images/'),samples_per_epoch=len(X), nb_epoch=1)

	# D = []
	# label = []
	# # here's a more "manual" example
	# for e in range(1):
	#     print 'Epoch', e
	#     batches = 0
	#     for X_batch, Y_batch in datagen.flow(X, y, batch_size=100):
	#     	D.append(X_batch)
	#     	label.append(Y_batch)
	#         batches += 1
	#         print('Batch_no:%s'%(batches))
	#         if batches >= len(X) / 100:
	#             # we need to break the loop by hand because
	#             # the generator loops indefinitely
	#             break

	# D = np.asarray(np.concatenate(D,0))
	# L = np.asarray(np.concatenate(label,0))
	# return D,L
	dim0,dim1,dim2,dim3 = X.shape
	flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
	print('Subtracting Mean....')
	flatX = flatX - np.mean(flatX,axis=(0),keepdims=True)
	print('Calculating Covariance Matrix....')
	sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
	print('Calculating SVD....')
	U, S, V = linalg.svd(sigma)
	print('Calculating zca_components....')
	pca_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)
	with open('pca_components_orig_aug_32_38432.pkl', 'w') as f:
		pickle.dump(pca_components,f)
	zca_components = np.reshape(np.dot(pca_components,X.T),(dim0,dim1,dim2,dim3))
	return X,y

def numpify(ip_dir,labelfile_path,luatype=True):
	D = []
	label = []
	id_ = []
	df = pd.read_csv(labelfile_path)
	df['Id'] = df['Id'].astype('|S12')
	for index,g in df.iterrows():
		filename = str(g[0])+'.jpg'
		if(index%500==0):
			print (index)
		img = scipy.ndimage.imread(os.path.join(ip_dir,filename))
		if luatype:
			D.append(np.moveaxis(img,2,0))
		else:
			D.append(img)
		label.append(int(g[1]))
		id_.append(str(g[0]))
	return np.asarray(D),np.asarray(label),id_
def numpy2img(X,y,op_dir):
	if(not os.path.exists(op_dir)):
		os.makedirs(op_dir)
	code = 0
	for i in range(len(y)):
		scipy.misc.imsave(os.path.join(op_dir,'im_'+str(code)+'_'+str(label)+'.jpg'),X[i])
