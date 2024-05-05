import sys, os, h5py, bz2, pylab as plt, importlib, itertools, numpy as np, time, psutil
from tqdm.auto import tqdm, trange
from collections import OrderedDict
import healpy as hp
from sklearn.neighbors import BallTree
import tensorflow as tf
process = psutil.Process()
tf.config.run_functions_eagerly(False)
tf.config.set_soft_device_placement(False)

print('testing real space sphere convolution speed')
print('using batch-last order: n_channels, n_pix, batch_size')


# get healpix map

nside=1024
npix = hp.nside2npix(nside)
m = np.ones(npix)
cl = hp.alm2cl(hp.map2alm(m))
cl = 1 - np.arange(len(cl))*0.0001
np.random.seed(1234)
m = hp.synfast(cl, nside=nside, pixwin=True)
m = m_full = m - np.mean(m)
m = np.float32(m)


print(f'nside={nside} npix={npix}')
print(f'got heaplix map with sum={np.sum(m)} mean={np.mean(m)}')

sigma_arcmin = 10/2.3
n_sigma_support = 3
sigma_rad = sigma_arcmin/60/180*np.pi

m_smooth_healpy = hp.sphtfunc.smoothing(m, sigma=sigma_rad)
print(f'smoothed map healpy sum={np.sum(m_smooth_healpy)} mean={np.mean(m_smooth_healpy)}')

# get coordinates of pixels
lon, lat = hp.pix2ang(nside, ipix=np.arange(npix), lonlat=True)
theta = np.vstack([np.radians(lat), np.radians(lon)]).T

print('using smaller sphere area')
ind_base = hp.ang2pix(nside=1, theta=lon, phi=lat, lonlat=True)
select = (ind_base<2)
theta = theta[select,:]
m = m[select]
m_smooth_healpy = m_smooth_healpy[select]
npix_full = npix
npix = len(m)
print(f'new npix={npix} new area={npix/npix_full*44000}')
    
# get tree
# The first coordinate of each point is assumed to be the latitude, the second is the longitude, given in radians. 
tree = BallTree(theta, metric='haversine')

load_dists = False

if load_dists:

    dist_k = np.load(f'dist_k_nside{nside}.npy')
    inds_k = np.load(f'inds_k_nside{nside}.npy')
    max_neighbours = inds_k.shape[1]

    print(f'loaded dist_k={dist_k.shape} inds_k={inds_k.shape}')

else:

    print(f'creating tree for {len(theta)} pixels and radius {n_sigma_support*sigma_arcmin} arcmin')

    inds_r, dist_r = tree.query_radius(theta, r=sigma_rad*n_sigma_support, return_distance=True, sort_results=True)
    n_neighbours = [len(i) for i in inds_r]
    max_neighbours = np.max(n_neighbours)
    print(f'max_neighbours={max_neighbours}')
    theta_split = np.array_split(theta, 100)
    list_dist_k, list_inds_k = [], []
    for theta_ in tqdm(theta_split):
        dist_k, inds_k  = tree.query(theta_, k=max_neighbours, return_distance=True, sort_results=True)
        list_dist_k.append(dist_k)
        list_inds_k.append(inds_k)
    dist_k = np.concatenate(list_dist_k, axis=0)
    inds_k = np.concatenate(list_inds_k, axis=0)
    print(inds_k.shape, dist_k.shape)

    np.save(f'dist_k_nside{nside}.npy', dist_k)
    np.save(f'inds_k_nside{nside}.npy', inds_k)
    
    print(f'stored dist_k={dist_k.shape} inds_k={inds_k.shape}')




# kernel_func = lambda r: 1./np.sqrt(2.*np.pi*sigma_rad**2) * np.exp(-0.5 * r**2/sigma_rad**2) 
# kernel_func = lambda r: np.exp(-0.5/sigma_rad**2 * r**2) 
def kernel_func(r):
    from scipy.stats import multivariate_normal
    d = r.ravel()
    d = np.vstack([d, np.zeros_like(d)]).T    
    k = multivariate_normal(mean=[0,0], cov=np.eye(2)*sigma_rad**2).pdf(d)
    k = np.reshape(k, r.shape)
    k = k * hp.nside2pixarea(nside)
    return k


kernel = kernel_func(dist_k)
kernel = np.float32(kernel)
inds_k = np.int64(inds_k)

print(f'kernel size {kernel.nbytes/1e9:4.2f} GB dtype {kernel.dtype}')
print(f'index size {inds_k.nbytes/1e9:4.2f} GB dtype {inds_k.dtype} max_ind={np.max(inds_k)}')
print(f'single map tensor {npix * max_neighbours * 4/1e9:4.2f} GB')


n_channels = 8
batch_size = 52



print('===========================> loop channels and batch sparse-dense convolution ')

n_trials = 100

kernel_channels = []
map_channels_batch = []

with tf.device('cpu'):

    for i in range(n_channels):

        inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.tile(inds_r, [1, max_neighbours])
        inds_c = tf.constant(inds_k, dtype=tf.int64)
        ind_coo = tf.concat([tf.reshape(inds_r, [-1,1]), tf.reshape(inds_c, [-1,1])], axis=-1)
        val_kernel = tf.reshape(kernel, [-1])
        # m_batch = tf.concat([tf.expand_dims(m, axis=-1)]*batch_size, axis=-1)

        for j in range(batch_size):
            map_channels_batch.append(tf.expand_dims(m, axis=-1))

        sparse_kernel = tf.sparse.SparseTensor(indices=ind_coo,
                                      values=val_kernel,
                                      dense_shape=[npix, npix])
        sparse_kernel = tf.sparse.reorder(sparse_kernel)
        kernel_channels.append(sparse_kernel)

        print(f'========> channel {i}')
        print(f'ind_coo.shape={ind_coo.shape} ind_coo.size={np.array(ind_coo).nbytes/1e9:2.4f} GB val_kernel.shape={val_kernel.shape} val_kernel.size={np.array(val_kernel).nbytes/1e9:2.4f} GB')
        print('memory used {:2.4f} GB'.format(process.memory_info().rss/1e9))  # in bytes 


print(f'created {n_channels} sparse kernels with shape {kernel_channels[0].shape} and batched maps with size {map_channels_batch[0].shape}')

time_start = time.time()
with tf.device('gpu'):

    for j in range(n_trials):
        map_batch_conv = []
        for i in range(n_channels):
            m_conv = tf.sparse.sparse_dense_matmul(kernel_channels[i], map_channels_batch[i])
            map_batch_conv.append(m_conv)
        map_batch_conv = tf.stack(map_batch_conv)
time_elapsed = (time.time()-time_start)/n_trials
print(f'n_trials={n_trials} time per trial: {time_elapsed:2.6f} s')

m_smooth_conv = map_batch_conv[0]
print(f'smoothed map sparse-dense sum={np.sum(m_smooth_conv)} mean={np.mean(m_smooth_conv)}')




print('===========================> loop sparse-dense convolution')

n_trials = 100

kernel_channels = []
map_channels_batch = []




with tf.device('cpu'):

    for i in range(n_channels):

        inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.tile(inds_r, [1, max_neighbours])
        inds_c = tf.constant(inds_k, dtype=tf.int64)
        ind_coo = tf.concat([tf.reshape(inds_r, [-1,1]), tf.reshape(inds_c, [-1,1])], axis=-1)
        val_kernel = tf.reshape(kernel, [-1])
        m_batch = tf.concat([tf.expand_dims(m, axis=-1)]*batch_size, axis=-1)
        map_channels_batch.append(m_batch)

        sparse_kernel = tf.sparse.SparseTensor(indices=ind_coo,
                                      values=val_kernel,
                                      dense_shape=[npix, npix])
        sparse_kernel = tf.sparse.reorder(sparse_kernel)
        kernel_channels.append(sparse_kernel)

        print(f'========> channel {i}')
        print(f'ind_coo.shape={ind_coo.shape} ind_coo.size={np.array(ind_coo).nbytes/1e9:2.4f} GB val_kernel.shape={val_kernel.shape} val_kernel.size={np.array(val_kernel).nbytes/1e9:2.4f} GB')
        print('memory used {:2.4f} GB'.format(process.memory_info().rss/1e9))  # in bytes 


print(f'created {n_channels} sparse kernels with shape {kernel_channels[0].shape} and batched maps with size {map_channels_batch[0].shape}')

time_start = time.time()
with tf.device('gpu'):

    for j in range(n_trials):
        map_batch_conv = []
        for i in range(n_channels):
            m_conv = tf.sparse.sparse_dense_matmul(kernel_channels[i], map_channels_batch[i])
            map_batch_conv.append(m_conv)
        map_batch_conv = tf.stack(map_batch_conv)
time_elapsed = (time.time()-time_start)/n_trials
print(f'n_trials={n_trials} time per trial: {time_elapsed:2.6f} s')

m_smooth_conv = map_batch_conv[0]
print(f'smoothed map sparse-dense sum={np.sum(m_smooth_conv)} mean={np.mean(m_smooth_conv)}')

print('===========================> block matrix sparse-dense convolution')

ind_batch = []
val_batch = []
map_batch = []

with tf.device('cpu'):

    for i in trange(n_channels, desc='creating sparse kernels'):

        inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.tile(inds_r, [1, max_neighbours])
        inds_c = tf.constant(inds_k, dtype=tf.int64)
        ind_coo = tf.concat([tf.reshape(inds_r, [-1,1]), tf.reshape(inds_c, [-1,1])], axis=-1)
        ind_coo = ind_coo + i * npix # block-diag
        ind_batch.append(ind_coo)
        val_batch.append(tf.reshape(kernel, [-1]))

        m_batch = tf.concat([tf.expand_dims(m, axis=-1)]*batch_size, axis=-1)
        map_batch.append(m_batch)


    ind_batch = tf.concat(ind_batch, axis=0)
    val_batch = tf.concat(val_batch, axis=0)
    map_batch = tf.concat(map_batch, axis=0)


    sparse_kernel = tf.sparse.SparseTensor(indices=ind_batch,
                                           values=val_batch,
                                           dense_shape=[npix*n_channels, npix*n_channels])
    sparse_kernel = tf.sparse.reorder(sparse_kernel)

    print(f'created block-sparse index shape={ind_batch.shape} max_val={np.max(ind_batch)} size={np.array(ind_batch).nbytes/1e9:2.4f} GB')
    print('memory used {:2.4f} GB'.format(process.memory_info().rss/1e9))  # in bytes 


time_start = time.time()
with tf.device('gpu'):

    for j in range(n_trials):

        m_batch_conv = tf.sparse.sparse_dense_matmul(sparse_kernel, map_batch)
        m_batch_conv = tf.reshape(m_batch_conv, [n_channels, -1, batch_size])

time_elapsed = (time.time()-time_start)/n_trials
print(f'n_trials={n_trials} time per trial: {time_elapsed:2.6f} s')

m_smooth_conv = m_batch_conv[0,:,0]
print(f'smoothed map block sparse-dense sum={np.sum(m_smooth_conv)} mean={np.mean(m_smooth_conv)}')

def part_to_full(m_part):
    m_ = m_full*0
    m_[select] = m_part
    return m_

np.save('kernel.npy', kernel)
np.save('m.npy', part_to_full(m))
np.save('m_smooth_conv.npy', part_to_full(m_smooth_conv))
np.save('m_smooth_healpy.npy', part_to_full(m_smooth_healpy))

print('===========================> loop sparse-dense convolution, single_kernel x channels+batch ')

n_trials = 100

kernel_channels = []
map_channels_batch = []




with tf.device('cpu'):

    for i in range(n_channels):

        inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.tile(inds_r, [1, max_neighbours])
        inds_c = tf.constant(inds_k, dtype=tf.int64)
        ind_coo = tf.concat([tf.reshape(inds_r, [-1,1]), tf.reshape(inds_c, [-1,1])], axis=-1)
        val_kernel = tf.reshape(kernel, [-1])
        m_batch = tf.concat([tf.expand_dims(m, axis=-1)]*batch_size, axis=-1)
        map_channels_batch.append(m_batch)

        sparse_kernel = tf.sparse.SparseTensor(indices=ind_coo,
                                      values=val_kernel,
                                      dense_shape=[npix, npix])
        sparse_kernel = tf.sparse.reorder(sparse_kernel)
        kernel_channels.append(sparse_kernel)

        print(f'========> channel {i}')
        print(f'ind_coo.shape={ind_coo.shape} ind_coo.size={np.array(ind_coo).nbytes/1e9:2.4f} GB val_kernel.shape={val_kernel.shape} val_kernel.size={np.array(val_kernel).nbytes/1e9:2.4f} GB')
        print('memory used {:2.4f} GB'.format(process.memory_info().rss/1e9))  # in bytes 


print(f'created {n_channels} sparse kernels with shape {kernel_channels[0].shape} and batched maps with size {map_channels_batch[0].shape}')

time_start = time.time()
with tf.device('gpu'):

    for j in range(n_trials):
        map_batch_conv = []
        for i in range(n_channels):
            m_conv = tf.sparse.sparse_dense_matmul(kernel_channels[i], map_channels_batch[i])
            map_batch_conv.append(m_conv)
        map_batch_conv = tf.stack(map_batch_conv)
time_elapsed = (time.time()-time_start)/n_trials
print(f'n_trials={n_trials} time per trial: {time_elapsed:2.6f} s')

m_smooth_conv = map_batch_conv[0]
print(f'smoothed map sparse-dense sum={np.sum(m_smooth_conv)} mean={np.mean(m_smooth_conv)}')


import pudb; pudb.set_trace();
pass



print('===========================> healpy convolution CPU')

n_trials = 1

time_start = time.time()

for i in trange(n_channels, desc='smoothing channels'):
    for j in range(batch_size):
        m_smooth_healpy = hp.sphtfunc.smoothing(m_full, sigma=sigma_rad)

time_elapsed = (time.time()-time_start)/n_trials
print(f'n_trials={n_trials} time per trial: {time_elapsed:2.6f} s')













# print('===========================> fun_map sparse-dense cov')

# ind_batch = []
# val_batch = []
# map_batch = []
# for i in range(n_channels):

#     inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
#     inds_r = tf.expand_dims(inds_r, axis=-1)
#     inds_r = tf.tile(inds_r, [1, max_neighbours])
#     inds_c = tf.constant(inds_k, dtype=tf.int64)
#     ind_coo = tf.concat([tf.reshape(inds_r, [-1,1]), tf.reshape(inds_c, [-1,1])], axis=-1)
#     ind_coo = tf.concat([tf.ones( [len(ind_coo), 1], dtype=tf.int64)*i, ind_coo], axis=1)
#     ind_batch.append(ind_coo)
#     val_batch.append(tf.reshape(kernel, [-1]))
#     map_batch.append(tf.expand_dims(m, axis=0))

# val_batch = tf.concat(val_batch, axis=0)
# ind_batch = tf.concat(ind_batch, axis=0)
# map_batch = tf.concat(map_batch, axis=0)

# sparse_kernel = tf.sparse.SparseTensor(indices=ind_batch,
#                               values=val_batch,
#                               dense_shape=[n_channels, npix, npix])
# sparse_kernel = tf.sparse.reorder(sparse_kernel)

# # https://stackoverflow.com/questions/42892347/can-i-apply-tf-map-fn-to-multiple-inputs-outputs
# # class Vehicle(tf.experimental.BatchableExtensionType):
# #   top_speed: tf.sparse.SparseTensor
# #   mpg: tf.Tensor
# # tensor_dense = tf.ones((3, 2, 1), dtype=tf.float32)
# # tensor_sparse = tf.sparse.SparseTensor(indices=[[0, 0, 0], [1, 1, 0]], values=tf.constant([1, 1], dtype=tf.float32), dense_shape=[3, 2, 2])
# # batch = Vehicle(tensor_sparse, tensor_dense)
# # tf.map_fn(lambda vehicle: tf.sparse.sparse_dense_matmul(vehicle.top_speed, vehicle.mpg), batch, fn_output_signature=tf.float32)


# class BatchConvData(tf.experimental.BatchableExtensionType):
#   k: tf.sparse.SparseTensor
#   m: tf.Tensor
# map_batch = tf.expand_dims(map_batch, axis=-1)
# batch = BatchConvData(sparse_kernel, map_batch)

# n_trials = 10

# time_start = time.time()
# with tf.device('gpu'):
    
#     for i in range(n_trials):
#         tf.map_fn(lambda x: tf.sparse.sparse_dense_matmul(x.k, x.m), batch, fn_output_signature=tf.float64)

# print(f'time: {time.time()-time_start:2.4f} s')


# import pudb; pudb.set_trace();
# pass







# print('===========================> full rank conv')

# n_channels = 8

# ind_batch = []
# val_batch = []
# map_batch = []
# for i in range(n_channels):

#     inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
#     inds_r = tf.expand_dims(inds_r, axis=-1)
#     inds_r = tf.tile(inds_r, [1, max_neighbours])
#     inds_c = tf.constant(inds_k, dtype=tf.int64)
#     ind_coo = tf.concat([tf.reshape(inds_r, [-1,1]), tf.reshape(inds_c, [-1,1])], axis=-1)
#     ind_coo = tf.concat([tf.ones( [len(ind_coo), 1], dtype=tf.int64)*i, ind_coo], axis=1)
#     ind_batch.append(ind_coo)
#     val_batch.append(tf.reshape(kernel, [-1]))
#     map_batch.append(tf.expand_dims(m, axis=0))

# val_batch = tf.concat(val_batch, axis=0)
# ind_batch = tf.concat(ind_batch, axis=0)
# map_batch = tf.concat(map_batch, axis=0)

# sparse_kernel = tf.sparse.SparseTensor(indices=ind_batch,
#                               values=val_batch,
#                               dense_shape=[n_channels, npix, npix])


# time_start = time.time()
# map_batch_convolved = tf.sparse.reduce_sum(sparse_kernel * map_batch[:, None, :], axis=-1)
# print(f'time: {time.time()-time_start:2.2f} s')
