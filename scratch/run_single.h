#pragma once

#include "common.h"

inline void test_single_solve(const fs::path &dir, json args, const int p, const Options &opt)
{
	const std::string tag = "no_refinement_" + std::to_string(p);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);

	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, args);
	visit_nc_mesh(*m, [&](auto &nc) { nc.refine(opt.base_refs, 0.5); nc.prepare_mesh(); });

	mesh::Mesh *mesh_ptr = m.get();
	state.variational_formulation->set_mesh(std::move(m));

	Eigen::MatrixXd sol;
	state.solve(sol);
	disable_fp_traps();

	io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
	if (stats.num_dofs == 0) // the nonlinear forms do not set it
		stats.num_dofs = sol.rows();
	std::cout << tag << ": n_el " << mesh_ptr->n_elements() << ", dofs " << stats.num_dofs
			  << ", L2 " << stats.l2_err << ", H1semi " << stats.h1_semi_err << std::endl;

	dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + ".vtu"));
	state.variational_formulation->export_data(sol);
}
