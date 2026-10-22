#pragma once

#include "common.h"

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/mesh/NodeCorrespondence.hpp>

// ======================================================================
// Coarse-vs-fine error estimator (scalar problems, polynomial degree 1 or 2)
//
//   1. copy the current coarse mesh into a second State and refine it uniformly once
//   2. solve on that fine mesh
//   3. match every coarse node with the fine node at the same place, through the
//      mesh refinement map (indices only, no positions)
//   4. d = u_fine - u_coarse at every matched node
//   5. eta_K^2 = |K| * mean over the nodes of K of d^2
//
// Needs the PolyFEM changes in polyfem_refinement_map.patch:
//   Mesh::refinement_map_to, mesh::node_correspondence, OutputSpace::mesh_nodes.
// ======================================================================

// DOF ids of every element, read from the per-element bases of the output space.
// These are the field bases when the space is iso-parametric, which the caller checks
// by comparing the DOF count with the solution size.
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

// Area (2D) or volume (3D) of every element.
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

// The State used for the fine solves. Initialising a State (state.init + load_mesh) is slow, so it
// is done once, on the first call; later calls only swap in the new mesh with set_mesh.
struct FineSolver
{
	State state;
	bool ready = false;
};

// Returns eta_K^2 for every element of the coarse mesh of `coarse_state`.
//   coarse_sol : solution of coarse_state, on its current mesh
//   args       : the args coarse_state was initialised with
inline Eigen::VectorXd error_estimation_simple(State &coarse_state, const Eigen::MatrixXd &coarse_sol, const json &args, FineSolver &fine)
{
	const io::OutputSpace cs = coarse_state.variational_formulation->output_space();
	if (!cs.mesh || !cs.mesh_nodes)
		log_and_throw_error("error_estimation_simple: the formulation does not expose its mesh nodes.");

	// fine problem: copy of the current coarse mesh, refined uniformly once
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

	const io::OutputSpace fs = fine_state.variational_formulation->output_space();
	if (!fs.mesh || !fs.mesh_nodes)
		log_and_throw_error("error_estimation_simple: the fine formulation does not expose its mesh nodes.");

	// coarse node -> fine node, from the refinement of the mesh
	const mesh::RefinementMap map = cs.mesh->refinement_map_to(*fs.mesh);
	const std::vector<int> fine_node = mesh::node_correspondence(*cs.mesh_nodes, *fs.mesh_nodes, map);

	if (static_cast<int>(fine_node.size()) != coarse_sol.rows() || fs.mesh_nodes->n_nodes() != fine_sol.rows())
		log_and_throw_error("error_estimation_simple: node count differs from the solution size (vector-valued problem?).");

	// difference at every coarse node that has a fine counterpart
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

	// aggregate per coarse element
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

// ======================================================================
// Uniform-refinement estimator, integrated over each coarse element (scalar problems, any degree)
//
//   1. copy the coarse mesh, refine it uniformly once, solve on it (same FineSolver as above)
//   2. for every coarse element K and every fine element f inside K, evaluate both solutions
//      at the quadrature points of f (the coarse one through K's own bases)
//   3. eta_K^2 = integral over K of (u_fine - u_coarse)^2           (L2)
//           or = integral over K of |grad u_fine - grad u_coarse|^2 (H1 semi)
//
// Unlike error_estimation_simple, this does not sample the difference at the coarse nodes only:
// the FE error is smallest at the nodes and largest between them, so nodal samples miss most of it.
// It also evaluates the coarse solution through the field bases, which differ from the geometry
// bases (and their numbering) when the space is not isoparametric, e.g. P2 on a straight mesh.
// Assumes straight (affine) simplices, which is what NC meshes are.
// ======================================================================

enum class EstimatorNorm
{
	L2,
	H1Semi
};

inline Eigen::VectorXd estimate_by_uniform_refinement(State &coarse_state, const Eigen::MatrixXd &coarse_sol, const json &args,
													  FineSolver &fine, const EstimatorNorm norm = EstimatorNorm::H1Semi)
{
	const io::OutputSpace cs = coarse_state.variational_formulation->output_space();
	if (!cs.mesh || !cs.bases || !cs.geometry_bases)
		log_and_throw_error("estimate_by_uniform_refinement: solve the coarse problem first.");

	// fine problem: copy of the current coarse mesh, refined uniformly once
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

	const io::OutputSpace fs = fine_state.variational_formulation->output_space();
	if (!fs.mesh || !fs.bases || !fs.geometry_bases)
		log_and_throw_error("estimate_by_uniform_refinement: the fine formulation has no bases.");

	const mesh::Mesh &cm = *cs.mesh;
	const mesh::Mesh &fm = *fs.mesh;
	const mesh::RefinementMap map = cm.refinement_map_to(fm);
	const int dim = cm.dimension();

	// vertices of the reference simplex: 0, e_1, ..., e_dim
	Eigen::MatrixXd ref_corners = Eigen::MatrixXd::Zero(dim + 1, dim);
	for (int i = 0; i < dim; ++i)
		ref_corners(i + 1, i) = 1;

	Eigen::VectorXd eta2 = Eigen::VectorXd::Zero(cm.n_elements());
	assembler::ElementAssemblyValues vals;
	Eigen::MatrixXd corners, fine_val, fine_grad, coarse_val, coarse_grad;

	for (int K = 0; K < cm.n_elements(); ++K)
	{
		// affine map of K: x = x0 + J xi, inverted to get the coarse reference coordinates
		(*cs.geometry_bases)[K].eval_geom_mapping(ref_corners, corners);
		const Eigen::RowVectorXd x0 = corners.row(0);
		Eigen::MatrixXd J(dim, dim);
		for (int i = 0; i < dim; ++i)
			J.col(i) = (corners.row(i + 1) - x0).transpose();
		const Eigen::PartialPivLU<Eigen::MatrixXd> J_lu(J);

		for (const int f : map.children[K])
		{
			vals.compute(f, fm.is_volume(), (*fs.bases)[f], (*fs.geometry_bases)[f]);
			io::Evaluator::interpolate_at_local_vals(f, dim, 1, vals, fine_sol, fine_val, fine_grad);

			// fine quadrature points (physical) -> reference coordinates of K
			Eigen::MatrixXd xi(vals.val.rows(), dim);
			for (int q = 0; q < vals.val.rows(); ++q)
				xi.row(q) = J_lu.solve((vals.val.row(q) - x0).transpose()).transpose();
			io::Evaluator::interpolate_at_local_vals(cm, 1, *cs.bases, *cs.geometry_bases, K, xi, coarse_sol, coarse_val, coarse_grad);

			const Eigen::ArrayXd diff2 = norm == EstimatorNorm::L2
											 ? (fine_val - coarse_val).rowwise().squaredNorm().array()
											 : (fine_grad - coarse_grad).rowwise().squaredNorm().array();
			eta2(K) += (diff2 * vals.det.array() * vals.quadrature.weights.array()).sum();
		}
	}
	return eta2;
}
