#include "densecrf.h"

class DenseCRFWrapper {
	public:
		DenseCRFWrapper(int npixels, int nlabels);
		virtual ~DenseCRFWrapper();

		void set_unary_energy(float* unary_costs_ptr);

		void add_pairwise_energy(float* pairwise_costs_ptr,
				float* features_ptr, int nfeatures);

		void map(int n_iters, int* result);

		int npixels();
		int nlabels();

	private:
		DenseCRF* m_crf;
		int m_npixels;
		int m_nlabels;
};

class DenseCRFWrapper2D {
	public:
		DenseCRFWrapper2D(int width, int height, int nlabels);
		virtual ~DenseCRFWrapper2D();

		void set_unary_energy(float* unary_costs_ptr);

		void addPairwiseGaussian(
			float sx,
			float sy,
			float potts_weight );


		void addPairwiseBilateral(
			float sx, float sy,
			float sr, float sg, float sb,
			const unsigned char * im,
			float potts_weight);

		void map(int n_iters, int* result);

		int width();
		int height();
		int nlabels();

	private:
		DenseCRF2D* m_crf;
		int m_width;
		int m_height;
		int m_nlabels;
};
