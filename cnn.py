import numpy as np 
def zero_pad(X,pad):
	'''
	Argument:
	X:-python numpy array of shape (m,n_H,n_W,n_C) representing a batch of m images
	pad:-integer, amount of padding around each image on vertical and horizontal demensions
    Returns:
    X_pad:-padded image of shape (m,n_H+2*pad,n_W+2*pad,n_C)
	'''
	X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
	return X_pad
def conv_single_step(a_slice_prev,W,b):
	'''
	Argument:
	a_slice_prev:-slice of input data of shape (f,f,n_C_prev)
	W:-Weight parameters contained in a window-matrix of shape(f,f,n_C_prev)
	b:-Bias parameters contained in a window - matrix of shape (1,1,1)
	Returns:
	Z:- A scaler value, result of convolving the sliding window (W,b) on a slice x of the input data
	'''
	#elementwise product between a_slice_prev and W
	s=a_slice_prev*W
	#sum over all the entries of the volume s.
	Z=np.sum(s)
	#Add bias b to Z.
	Z=Z+float(b)
	return Z
def conv_forward(A_prev,W,b,hparameters):
	'''
	Argument:
	A_prev:-output activation of the previous layer, numpy array of shape (m,n_H_prev,n_W_prev,n_C_prev)
	W:- Weights, numpy array of shape (f,f,n_C_prev,n_C)
	b:-Biases, numpy array of shape (1,1,1,n_C)
	hparameters: python dictionary containing "stride" and "pad"
	Returns:
	Z:-conv output, numpy array of shape (m,n_H,n_W,n_C)
	cache:- cache of values needed for the backpropogation
	'''
	# Retrieve dimensions from A_prev's shape
	(m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
	#Retrieve information from W's shape
	(f,f,n_C_prev,n_C)=W.shape
	# Retrieve information from hyperoarameters
	stride=hparameters['stride']
	pad=hparameters['pad']
	# Compute the dimension of the CONV output volume
	n_H=int(((n_H_prev+2*pad-f)/stride)+1)
	n_W=int(((n_H_prev+2*pad-f)/stride)+1)
	# Initialize the output volumn Z with zeros
	Z=np.zeros((m,n_H,n_W,n_C))
	# Create A_prev_pad by padding A_prev
	A_prev_pad=zero_pad(A_prev,pad)
	for i in range(m): #loop over the batch of the training example
		a_prev_pad=A_prev[i] # select ith training example's padded activation
		for h in range(n_H): # loop over vertical axis of the output volumn
			for w in range(n_W): # loop over horizontal axis of the output volumn
				for c in range(n_C): # loop over channels(=#filtters) of the output volumn
					# Find the corners of the current slice
					vert_start=stride*h
					vert_end=vert_start+f
					horiz_start=stride*w
					horiz_end=horiz_start+f
					# Use the corners to define the (3D) slice of a_prev_pad
					a_slice_prev=A_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
					# Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
					Z[i,h,W,c]=conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])

	assert(Z.shape==(m,n_H,n_W,n_C))
	cache=(A_prev,W,b,hparameters)
	return Z,cache

def pool_forward(A_prev,hparameters,mode="max"):
	'''
	Arguments:
	A_prev:-Input data of shape (m,n_H_prev,n_W_prev,n_C_prev)
	hparameters:- python dictionary containing "f" and "stride"
	mode:- the pooling mode would you like to use, defined as a string ("max" or "average")
	Returns:
	A:- output of the pool layer, a numpy array of shape (m,n_H,n_W,n_C)
	cache:-cache used in the backward pass of the pooling layer
	'''
	# Reterive dimensions from the input A_prev
	(m,n_H_prev,n_W_prev,n_C_prev)=A_prev.shape
	# Reterive information from the dictionary
	f=hparameters["f"]
	stride=hparameters["stride"]
	# Define the dimension of the output
	n_H=int(((n_H_prev-f)/stride)+1)
	n_W=int(((n_H_prev-f)/stride)+1)
	n_C=n_C_prev

	# Initialize the output volumn Z with zeros
	A=np.zeros((m,n_H,n_W,n_C))
	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):
					vert_start=stride*h
					vert_end=vert_start+f
					horiz_start=stride+w
					horiz_end=horiz_start+f
					a_slice_prev=A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]

					if(mode=="max"):
						A[i,n_H,n_W,n_C]=np.max(a_slice_prev)
					elif(mode=="average"):
						A[i,n_H,n_W,n_C]=np.mean(a_slice_prev)

	assert(A.shape==(m,n_H,n_W,n_C))
	cache = (A_prev, hparameters)
	return A,cache






