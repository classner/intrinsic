#include <Eigen/Core>
#include "densecrf.h"
#include "densecrf_wrapper.h"

DenseCRFWrapper::DenseCRFWrapper(int npixels, int nlabels)
: m_npixels(npixels), m_nlabels(nlabels) {
	m_crf = new DenseCRF(npixels, nlabels);
}

DenseCRFWrapper::~DenseCRFWrapper() {
	delete m_crf;
}

int DenseCRFWrapper::npixels() { return m_npixels; }
int DenseCRFWrapper::nlabels() { return m_nlabels; }

void DenseCRFWrapper::add_pairwise_energy(float* pairwise_costs_ptr, float* features_ptr, int nfeatures) {
	m_crf->addPairwiseEnergy(
		Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels),
		new MatrixCompatibility(
			Eigen::Map<const Eigen::MatrixXf>(pairwise_costs_ptr, m_nlabels, m_nlabels)
		),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC
	);
}

void DenseCRFWrapper::set_unary_energy(float* unary_costs_ptr) {
	m_crf->setUnaryEnergy(
		Eigen::Map<const Eigen::MatrixXf>(
			unary_costs_ptr, m_nlabels, m_npixels)
	);
}

void DenseCRFWrapper::map(int n_iters, int* labels) {
	VectorXs labels_vec = m_crf->map(n_iters);
	for (int i = 0; i < m_npixels; i ++)
		labels[i] = labels_vec(i);
}

///////////////////////////////////////////////////////////////////////////////
// 2D
DenseCRFWrapper2D::DenseCRFWrapper2D(int width, int height, int nlabels)
: m_width(width), m_height(height), m_nlabels(nlabels) {
	m_crf = new DenseCRF2D(width, height, nlabels);
}

DenseCRFWrapper2D::~DenseCRFWrapper2D() {
	delete m_crf;
}

int DenseCRFWrapper2D::width() { return m_width; }
int DenseCRFWrapper2D::height() { return m_height; }
int DenseCRFWrapper2D::nlabels() { return m_nlabels; }

void DenseCRFWrapper2D::set_unary_energy(float* unary_costs_ptr) {
	m_crf->setUnaryEnergy(
		Eigen::Map<const Eigen::MatrixXf>(
			unary_costs_ptr, m_nlabels, m_height * m_width)
	);
}

void DenseCRFWrapper2D::addPairwiseGaussian(float sx, float sy, float potts_weight) {
	m_crf->addPairwiseGaussian(
		sx,
		sy,
		new PottsCompatibility(potts_weight),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC);
}

void DenseCRFWrapper2D::addPairwiseBilateral(float sx, float sy, float sr, float sg, float sb, const unsigned char *im, float potts_weight) {
	m_crf->addPairwiseBilateral(
		sx,
		sy,
		sr,
		sg,
		sb,
		im,
		new PottsCompatibility(potts_weight),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC);
}

void DenseCRFWrapper2D::map(int n_iters, int* labels) {
	VectorXs labels_vec = m_crf->map(n_iters);
	for (int i = 0; i < m_height * m_width; i ++)
		labels[i] = labels_vec(i);
}
