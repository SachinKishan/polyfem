#pragma once

#include "common.h"

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/mesh/NodeCorrespondence.hpp>

// Needs polyfem_refinement_map.patch: Mesh::refinement_map_to, mesh::node_correspondence, OutputSpace::mesh_nodes.

inline std::vector<std::vector<int>> element_dofs(const io::OutputSpace &space)
{
	if (!space.geometry_bases)
		log_and_throw_error("element_dofs(): the formulation has no bases yet, solve first.");

	const std::vector<basis::ElementBases> &bases = *space.geometry_bases;
	std::vector<std::vector<int>> out(bases.size());
	for (size_t e = 0; e < bases.size(); ++e)
		for (const auto &b : bases[e].bases)
			for (const auto &g : b.global())
				out[e].push_back(g.index);
	return out;
}

//cakculate size of every element
inline Eigen::VectorXd element_measures(const mesh::Mesh &m)
{
	const int dim = m.dimension();
	Eigen::VectorXd out(m.n_elements());
	for (int e = 0; e < m.n_elements(); ++e)
	{
		Eigen::Vector3d p[4];
		for (int i = 0; i <= dim; ++i)
		{
			p[i].setZero();
			p[i].head(dim) = m.point(m.element_vertex(e, i)).transpose();
		}
		if (dim == 2)
			out(e) = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).norm();
		else
			out(e) = std::abs((p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0]))) / 6.0;
	}
	return out;
}

// state.init + load_mesh is slow, so it is done once, on the first call
struct FineSolver
{
	State state;
	bool ready = false;
};

inline Eigen::VectorXd error_estimation_simple(State &coarse_state, const Eigen::MatrixXd &coarse_sol, const json &args, FineSolver &fine)
{
	const io::OutputSpace cs = coarse_state.variational_formulation->output_space();
	if (!cs.mesh || !cs.mesh_nodes)
		log_and_throw_error("error_estimation_simple: the formulation does not expose its mesh nodes.");

	State &fine_state = fine.state;
	if (!fine.ready)
	{
		load_nc_mesh(fine_state, args); // only to initialise fine_state, the mesh it returns is dropped
		fine.ready = true;
	}
	std::unique_ptr<mesh::Mesh> fine_mesh = cs.mesh->copy();
	fine_mesh->refine(1, 0.5);
	fine_mesh->prepare_mesh();
	fine_state.variational_formulation->set_mesh(std::move(fine_mesh));

	Eigen::MatrixXd fine_sol;
	fine_state.solve(fine_sol);
	disable_fp_traps();

	const io::OutputSpace fs = fine_state.variational_formulation->output_space();
	if (!fs.mesh || !fs.mesh_nodes)
		log_and_throw_error("error_estimation_simple: the fine formulation does not expose its mesh nodes.");

	const mesh::RefinementMap map = cs.mesh->refinement_map_to(*fs.mesh);
	const std::vector<int> fine_node = mesh::node_correspondence(*cs.mesh_nodes, *fs.mesh_nodes, map);

	if (static_cast<int>(fine_node.size()) != coarse_sol.rows() || fs.mesh_nodes->n_nodes() != fine_sol.rows())
		log_and_throw_error("error_estimation_simple: node count differs from the solution size (vector-valued problem?).");

	Eigen::VectorXd d2 = Eigen::VectorXd::Zero(coarse_sol.rows());
	std::vector<bool> matched(coarse_sol.rows(), false);
	int n_unmatched = 0;
	for (int n = 0; n < coarse_sol.rows(); ++n)
	{
		if (fine_node[n] < 0)
		{
			++n_unmatched;
			continue;
		}
		const double d = fine_sol(fine_node[n], 0) - coarse_sol(n, 0);
		d2(n) = d * d;
		matched[n] = true;
	}
	if (n_unmatched > 0)
		std::cout << "error_estimation_simple: " << n_unmatched << " of " << coarse_sol.rows()
				  << " coarse nodes have no fine node and are skipped" << std::endl;

	const std::vector<std::vector<int>> dofs = element_dofs(cs);
	if (static_cast<int>(dofs.size()) != cs.mesh->n_elements())
		log_and_throw_error("error_estimation_simple: number of element bases differs from the number of mesh elements.");

	const Eigen::VectorXd measure = element_measures(*cs.mesh);
	Eigen::VectorXd eta2 = Eigen::VectorXd::Zero(cs.mesh->n_elements());
	for (int e = 0; e < cs.mesh->n_elements(); ++e)
	{
		double sum = 0;
		int count = 0;
		for (int n : dofs[e])
			if (matched[n])
			{
				sum += d2(n);
				++count;
			}
		if (count > 0)
			eta2(e) = measure(e) * sum / count;
	}
	return eta2;
}

// Uniform-refinement estimator, used by the runs. Integrates over each coarse element, because the error is
// smallest at the nodes and nodal samples miss most of it. Assumes straight (affine) simplices.

inline int n_field_bases(const std::vector<basis::ElementBases> &bases)
{
	int n = 0;
	for (const auto &eb : bases)
		for (const auto &b : eb.bases)
			for (const auto &g : b.global())
				n = std::max(n, g.index + 1);
	return n;
}

enum class EstimatorNorm
{
	L2,
	H1Semi
};

struct EstimatorResult
{
	Eigen::VectorXd l2;
	Eigen::VectorXd h1_semi;
	const Eigen::VectorXd &operator[](const EstimatorNorm norm) const { return norm == EstimatorNorm::L2 ? l2 : h1_semi; }
};

inline EstimatorResult estimate_by_uniform_refinement_both(State &coarse_state, const Eigen::MatrixXd &coarse_sol, const json &args,
														   FineSolver &fine)
{
	const io::OutputSpace cs = coarse_state.variational_formulation->output_space();
	if (!cs.mesh || !cs.bases || !cs.geometry_bases)
		log_and_throw_error("estimate_by_uniform_refinement: solve the coarse problem first.");

	State &fine_state = fine.state;
	if (!fine.ready)
	{
		load_nc_mesh(fine_state, args); // only to initialise fine_state, the mesh it returns is dropped
		fine.ready = true;
	}
	std::unique_ptr<mesh::Mesh> fine_mesh = cs.mesh->copy();
	fine_mesh->refine(1, 0.5);
	fine_mesh->prepare_mesh();
	fine_state.variational_formulation->set_mesh(std::move(fine_mesh));

	Eigen::MatrixXd fine_sol;
	fine_state.solve(fine_sol);
	disable_fp_traps();

	const io::OutputSpace fs = fine_state.variational_formulation->output_space();
	if (!fs.mesh || !fs.bases || !fs.geometry_bases)
		log_and_throw_error("estimate_by_uniform_refinement: the fine formulation has no bases.");

	const mesh::Mesh &cm = *cs.mesh;
	const mesh::Mesh &fm = *fs.mesh;
	const mesh::RefinementMap map = cm.refinement_map_to(fm);
	const int dim = cm.dimension();
	// components per node: 1 for scalar problems, dim for elasticity (solution is flattened node-major)
	const int actual_dim = int(coarse_sol.rows()) / n_field_bases(*cs.bases);
	if (actual_dim * n_field_bases(*cs.bases) != coarse_sol.rows() || actual_dim * n_field_bases(*fs.bases) != fine_sol.rows())
		log_and_throw_error("estimate_by_uniform_refinement: solution size is not a multiple of the number of bases.");

	Eigen::MatrixXd ref_corners = Eigen::MatrixXd::Zero(dim + 1, dim);
	for (int i = 0; i < dim; ++i)
		ref_corners(i + 1, i) = 1;

	EstimatorResult eta2{Eigen::VectorXd::Zero(cm.n_elements()), Eigen::VectorXd::Zero(cm.n_elements())};
	assembler::ElementAssemblyValues vals;
	Eigen::MatrixXd corners, fine_val, fine_grad, coarse_val, coarse_grad;

	for (int K = 0; K < cm.n_elements(); ++K)
	{
		(*cs.geometry_bases)[K].eval_geom_mapping(ref_corners, corners);
		const Eigen::RowVectorXd x0 = corners.row(0);
		Eigen::MatrixXd J(dim, dim);
		for (int i = 0; i < dim; ++i)
			J.col(i) = (corners.row(i + 1) - x0).transpose();
		const Eigen::PartialPivLU<Eigen::MatrixXd> J_lu(J);

		for (const int f : map.children[K])
		{
			vals.compute(f, fm.is_volume(), (*fs.bases)[f], (*fs.geometry_bases)[f]);
			io::Evaluator::interpolate_at_local_vals(f, dim, actual_dim, vals, fine_sol, fine_val, fine_grad);

			Eigen::MatrixXd xi(vals.val.rows(), dim);
			for (int q = 0; q < vals.val.rows(); ++q)
				xi.row(q) = J_lu.solve((vals.val.row(q) - x0).transpose()).transpose();
			io::Evaluator::interpolate_at_local_vals(cm, actual_dim, *cs.bases, *cs.geometry_bases, K, xi, coarse_sol, coarse_val, coarse_grad);

			const Eigen::ArrayXd w = vals.det.array() * vals.quadrature.weights.array();
			eta2.l2(K) += ((fine_val - coarse_val).rowwise().squaredNorm().array() * w).sum();
			eta2.h1_semi(K) += ((fine_grad - coarse_grad).rowwise().squaredNorm().array() * w).sum();
		}
	}
	return eta2;
}

inline Eigen::VectorXd estimate_by_uniform_refinement(State &coarse_state, const Eigen::MatrixXd &coarse_sol, const json &args,
													  FineSolver &fine, const EstimatorNorm norm = EstimatorNorm::H1Semi)
{
	return estimate_by_uniform_refinement_both(coarse_state, coarse_sol, args, fine)[norm];
}

enum class MarkScheme
{
	ErrorPercentage, // Doerfler
	ElementPercentage
};

// Doerfler marking
// mark elements who accumutively make up the theta fraction of magnitude of error
inline std::vector<int> mark_error_percentage(const Eigen::VectorXd &eta2, const double theta)
{
	std::vector<int> idx(eta2.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return eta2(a) > eta2(b); });
	const double target = theta * eta2.sum();
	double acc = 0;
	std::vector<int> marked;
	for (int i : idx)
	{
		marked.push_back(i);
		if ((acc += eta2(i)) >= target)
			break;
	}
	return marked;
}

//mark theta fraction of elements based on descending order of magnitude of error
inline std::vector<int> mark_element_percentage(const Eigen::VectorXd &eta2, const double theta)
{
	std::vector<int> idx(eta2.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return eta2(a) > eta2(b); });
	const int target = std::max(1, int(theta * eta2.size())); // marks 1 element if number of elements is very esmall
	int acc = 0;
	std::vector<int> marked;
	for (int i : idx)
	{
		if (acc >= target)
			break;
		++acc;
		marked.push_back(i);
	}
	return marked;
}

inline std::vector<int> mark_elements(const Eigen::VectorXd &eta2, const double theta, const MarkScheme scheme)
{
	if (theta <= 0)
		return {};
	return scheme == MarkScheme::ErrorPercentage ? mark_error_percentage(eta2, theta) : mark_element_percentage(eta2, theta);
}
