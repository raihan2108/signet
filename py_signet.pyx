import cython
import numpy as np
cimport numpy as np

from libcpp.map cimport map
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "cpp_signet.cpp":
    void signet(long* source, long* target, long* weight, double* embed_vertex, double* embed_context,
          map[long, vector[long]] samp, map[long, vector[long]] samn,
          long n_vertices, long n_edges, long n_dims, double init_rho, long n_iterations,
          long n_negatives, long n_order, long n_samples, bool is_sample)


@cython.boundscheck(False)
@cython.wraparound(False)

def py_signet(np.ndarray[long, ndim=1, mode="c"] source, np.ndarray[long, ndim=1, mode="c"] target,
            np.ndarray[long, ndim=1, mode="c"] weight, np.ndarray[double, ndim=1, mode="c"]embed_vertex,
            np.ndarray[double, ndim=1, mode="c"]embed_context, map[long, vector[long]] samp, map[long, vector[long]] samn,
            long n_vertices, long n_edges, long n_dims, double init_rho, long n_iter, long n_negatives, long order,
            long n_samples, bool is_sample):
    signet(&source[0], &target[0], &weight[0], &embed_vertex[0], &embed_context[0], samp, samn,
          n_vertices, n_edges, n_dims, init_rho, n_iter, n_negatives, order, n_samples, is_sample)