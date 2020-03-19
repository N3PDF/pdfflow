from pdfflow.subgrid import subgrid
import tensorflow as tf
import re
import numpy as np

float64 = tf.float64
int64 = tf.int64

def load_Data(fname):
    #Reads pdf from file and retrieves a list of grids
    #Each grid is a tuple containing numpy arrays (x,Q2, flavours, pdf)
    f = open(fname)

    n = []
    count = 0
    for line in f:
        if re.match('---', line):
            n += [count]

        count += 1
    f.close()

    grids = []
    for i in range(len(n)-1):
        x = np.loadtxt(fname, skiprows=(n[i]+1), max_rows=1)
        Q2 = np.loadtxt(fname, skiprows=(n[i]+2), max_rows=1)
        flav = np.loadtxt(fname, skiprows=(n[i]+3), max_rows=1)
        grid = np.loadtxt(fname, skiprows=(n[i]+4), max_rows=(n[i+1]-n[i]-4))

        grids += [(x,Q2,flav,grid)]

    return grids


class mkPDF:
    def __init__(self, fname, dirname='./local/share/LHAPDF/'):
        '''
        fname: string, must be in the format: '<set_name>/<set member number>'
        '''
        self.dirname = dirname
        f = fname.split('/')

        self.fname = self.dirname+'%s/%s_%s.dat'%(f[0],f[0],f[1].zfill(4))



        print('pdfflow loading ' + self.fname)
        grids = load_Data(self.fname)
        #[(x,Q2,flav,knots), ...]
        flav = list(map(lambda g: g[2] ,grids))
        for i in range(len(flav)-1):
            if not np.all(flav[i]  == flav[i+1]):
                print('Flavor schemes do not match across all the subgrids ---> algorithm will break !')


        self.subgrids = list(map(subgrid, grids))
        self.flavor_scheme = self.subgrids[0].flav


    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),tf.TensorSpec(shape=[None], dtype=float64), tf.TensorSpec(shape=[None], dtype=float64)])
    def _xfxQ2(self, u, aa_x, aa_Q2):

        a_x = tf.math.log(aa_x, name='logx')
        a_Q2 = tf.math.log(aa_Q2, name='logQ2')

        f_idx = tf.TensorArray(dtype=int64, size=0, dynamic_size=True, infer_shape=False, name='f_idx')
        f_f = tf.TensorArray(dtype=float64, size=0, dynamic_size=True, infer_shape=False, name='f_f')

        count = 0

        for i in range(len(self.subgrids)):
            p = self.subgrids[i]
            stripe = tf.math.logical_and(a_Q2 >= tf.math.log(p.Q2min), a_Q2 < tf.math.log(p.Q2max))


            in_x = tf.boolean_mask(a_x, stripe)
            in_Q2 = tf.boolean_mask(a_Q2, stripe)

            ff_idx = tf.cast(tf.where(stripe), dtype=int64)

            

            ff_f = p.interpolate(u, in_x, in_Q2)

            f_idx = f_idx.write(count, ff_idx)
            f_f = f_f.write(count, ff_f)

            count += 1

        f_idx = f_idx.concat()
        f_f = f_f.concat()

        # This will force recompilation as the shape of f_f is dynamic
        return tf.scatter_nd(f_idx, f_f, tf.shape(f_f, out_type=int64))

    def xfxQ2(self, PID, a_x, a_Q2):

        #must feed a mask for flavors to _xfxQ2
        #if PID is None, the mask is set to true everywhere
        #PID must be a list of PIDs
        if type(PID)==int:
            PID=[PID]
        
        PID = tf.expand_dims(tf.constant(PID, dtype=int64),-1)
        idx = tf.where(tf.equal(self.flavor_scheme, PID))[:,1]
        u, i = tf.unique(idx)

        f_f = self._xfxQ2(u, a_x, a_Q2).numpy()
        '''
        if asdict == True:
            res = {}
            for i in range(len(PID)):
                k = int(PID[i])
                col = int(tf.where(tf.equal(idx,i))[0,0])
                res[k] = f_f[:,col]
            return res
        '''

        f_f = tf.gather(f_f,i,axis=1)

        return tf.squeeze(f_f)

    def xfxQ2_allpid(self, a_x, a_Q2):
    	#return all the flavors
    	PID = self.flavor_scheme
    	return self.xfxQ2(PID, a_x, a_Q2)
