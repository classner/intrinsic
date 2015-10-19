# distutils: sources = src/densecrf_wrapper.cpp

cimport numpy as np

cdef extern from "include/densecrf_wrapper.h":
    cdef cppclass DenseCRFWrapper:
        DenseCRFWrapper(int, int) except +
        void set_unary_energy(float*)
        void add_pairwise_energy(float*, float*, int)
        void map(int, int*)
        int npixels()
        int nlabels()
    cdef cppclass DenseCRFWrapper2D:
        DenseCRFWrapper2D(int, int, int) except +
        void set_unary_energy(float*)
        void addPairwiseGaussian(float, float, float)
        void addPairwiseBilateral(float, float, float, float, float, const unsigned char*, float)
        void map(int, int*)
        int width()
        int height()
        int nlabels()

cdef class DenseCRF:
    cdef DenseCRFWrapper *thisptr

    def __cinit__(self, int npixels, int nlabels):
        self.thisptr = new DenseCRFWrapper(npixels, nlabels)

    def __dealloc__(self):
        del self.thisptr

    def set_unary_energy(self, float[:, ::1] unary_costs):
        if (unary_costs.shape[0] != self.thisptr.npixels() or
                unary_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid unary_costs shape")

        self.thisptr.set_unary_energy(&unary_costs[0, 0])

    def add_pairwise_energy(self, float[:, ::1] pairwise_costs,
                            float[:, ::1] features):
        if (pairwise_costs.shape[0] != self.thisptr.nlabels() or
                pairwise_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid pairwise_costs shape")
        if (features.shape[0] != self.thisptr.npixels()):
            raise ValueError("Invalid features shape")

        self.thisptr.add_pairwise_energy(
            &pairwise_costs[0, 0],
            &features[0, 0],
            features.shape[1]
        )

    def map(self, int n_iters=10):
        import numpy as np
        labels = np.empty(self.thisptr.npixels(), dtype=np.int32)
        cdef int[::1] labels_view = labels
        self.thisptr.map(n_iters, &labels_view[0])
        return labels


cdef class DenseCRF2D:
    cdef DenseCRFWrapper2D *thisptr

    def __cinit__(self, int width, int height, int nlabels):
        self.thisptr = new DenseCRFWrapper2D(width, height, nlabels)

    def __dealloc__(self):
        del self.thisptr

    def set_unary_energy(self, float[:, :, ::1] unary_costs):
        if (unary_costs.shape[0] != self.thisptr.height() or
            unary_costs.shape[1] != self.thisptr.width() or
            unary_costs.shape[2] != self.thisptr.nlabels()):
            raise ValueError("Invalid unary_costs shape")

        self.thisptr.set_unary_energy(&unary_costs[0, 0, 0])

    def add_pairwise_gaussian(self, float sx, float sy, float potts_weight):
        self.thisptr.addPairwiseGaussian(sx, sy, potts_weight)

    def add_pairwise_bilateral(self, float sx, float sy, float sr, float sg, float sb,
                               np.ndarray[np.uint8_t, ndim=3] im,
                               float potts_weight):
        self.thisptr.addPairwiseBilateral(
            sx, sy,
            sr, sg, sb,
            &im[0, 0, 0],
            potts_weight
        )

    def map(self, int n_iters=10):
        import numpy as np
        labels = np.empty((self.thisptr.height(), self.thisptr.width()), dtype=np.int32)
        cdef int[:, ::1] labels_view = labels
        self.thisptr.map(n_iters, &labels_view[0, 0])
        return labels
