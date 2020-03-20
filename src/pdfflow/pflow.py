from pdfflow.subgrid import Subgrid, act_on_empty
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

        self.subgrids = list(map(Subgrid, grids))
        self.flavor_scheme = tf.cast(self.subgrids[0].flav, dtype=tf.int64)

        # Generate switch cases

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=int64),tf.TensorSpec(shape=[None], dtype=float64), tf.TensorSpec(shape=[None], dtype=float64)])
    def _xfxQ2(self, u, aa_x, aa_Q2):

        a_x = tf.math.log(aa_x, name='logx')
        a_Q2 = tf.math.log(aa_Q2, name='logQ2')

        size = tf.shape(a_x)
        shape = tf.cast(tf.concat([size, tf.shape(u)], 0), int64)
        empty_fn = lambda: tf.constant(0.0, dtype=float64)

        count = 0
        l_idx = []
        res = tf.zeros(shape, dtype=float64)

        for subgrid in self.subgrids:
            # Check whether any points go through
            stripe = tf.math.logical_and(a_Q2 >= subgrid.log_q2min, a_Q2 < subgrid.log_q2max)
            f_idx = tf.where(stripe)

            def gen_fun():
                in_x = tf.boolean_mask(a_x, stripe)
                in_Q2 = tf.boolean_mask(a_Q2, stripe)
                ff_f = subgrid.interpolate(u, in_x, in_Q2)
                return tf.scatter_nd(f_idx, ff_f, shape)
            
            res += act_on_empty(f_idx, empty_fn, gen_fun)


        return res


    def xfxQ2(self, PID, a_x, a_Q2):

        #must feed a mask for flavors to _xfxQ2
        #if PID is None, the mask is set to true everywhere
        #PID must be a list of PIDs
        if type(PID)==int:
            PID=[PID]

        PID = tf.expand_dims(tf.constant(PID, dtype=int64),-1)
        idx = tf.where(tf.equal(self.flavor_scheme, PID))[:,1]
        u, i = tf.unique(idx)

        f_f = self._xfxQ2(u, a_x, a_Q2)
        f_f = tf.gather(f_f,i,axis=1)

        return tf.squeeze(f_f)

    def xfxQ2_allpid(self, a_x, a_Q2):
    	#return all the flavors
    	PID = self.flavor_scheme
    	return self.xfxQ2(PID, a_x, a_Q2)
