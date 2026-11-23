#pragma once

#include "common.h"
#include "timing.h"
#include "estimator.h"
#include "stopping.h"
#include "resume.h"

#include <polyfem/utils/Selection.hpp>

// Uses StopCriteria and Options from main.cpp, so it is included after them.

// CMesh2D/3D::refine keeps the old body and boundary ids, which no longer match: reset them as GeometryLoader does
inline void refine_standard_mesh(mesh::Mesh &m, const int n_refs, json args)
{
	if (n_refs <= 0)
		return;
	const int body_id = m.get_body_id(0);
	m.refine(n_refs, 0.5);
	m.prepare_mesh();
	m.set_body_ids(std::vector<int>(m.n_elements(), body_id));

	const json &geometry = first_geometry(args);
	if (!geometry.contains("surface_selection") || geometry["surface_selection"].is_null())
		return;
	utils::Selection::BBox bbox;
	m.bounding_box(bbox[0], bbox[1]);
	const io::FileSystemIO resources(args.contains("root_path") ? args["root_path"].get<std::string>() : std::string());
	const auto selections = utils::Selection::build_selections(geometry["surface_selection"], bbox, resources);
	bool boundary_only = true;
	for (const auto &s : selections)
		boundary_only &= s->boundary_only();

	m.compute_boundary_ids([&](const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p, bool is_boundary) {
		if (boundary_only && !is_boundary)
			return -1;
		for (const auto &s : selections)
		{
			if (s->boundary_only() && !is_boundary)
				continue;
			if (s->inside(p_id, vs, p))
				return s->id(p_id, vs, p);
		}
		return std::numeric_limits<int>::max(); // same default as GeometryLoader
	});
}

inline UniformResult test_uniform_refinement(const fs::path &dir, json args, const int p, const Options &opt,
											 const StopCriteria &stop, const bool non_conforming = true)
{
	const Method method = non_conforming ? Method::NCUniform : Method::Conforming;
	const std::string tag = method_name(method) + "_p" + std::to_string(p);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);

	// the estimator needs an NC mesh (refinement_map_to)
	const bool has_ref = opt.use_exact_error && has_reference(args);
	const bool use_estimator = !has_ref && non_conforming;
	if (!has_ref && !non_conforming)
		std::cout << tag << ": no output.reference and the estimator needs an NC mesh, errors are not computed" << std::endl;
	FineSolver fine;
	fine.warm_start = opt.fine_warm_start;

	if (opt.reuse_finished_runs)
	{
		const json done = read_completed(dir);
		if (done.contains(tag))
		{
			UniformResult r;
			r.first_err = json_err(done[tag]["first_err"]);
			r.last_err = json_err(done[tag]["last_err"]);
			std::cout << tag << ": already finished in " << dir << ", reusing it (err " << r.first_err
					  << " -> " << r.last_err << "; --fresh redoes it)" << std::endl;
			return r;
		}
	}

	StopCriteria uniform_stop = stop;
	if (stop.mode == StopCriteria::Mode::MatchUniform)
	{
		uniform_stop.absolute_tol = stop.uniform_absolute_tol;
		uniform_stop.relative_tol = stop.uniform_relative_tol;
	}
	UniformResult result;
	std::string final_reason;

	Stopwatch sw;
	State state;
	std::unique_ptr<mesh::Mesh> m = load_fem_mesh(state, args, non_conforming);
	std::ofstream(dir / "resolved_args.json") << state.args.dump(2) << "\n";
	if (non_conforming)
		visit_nc_mesh(*m, [&](auto &nc) { nc.refine(opt.base_refs, 0.5); nc.prepare_mesh(); });
	else
		refine_standard_mesh(*m, opt.base_refs, args);

	mesh::Mesh *mesh_ptr = m.get();
	state.variational_formulation->set_mesh(std::move(m));
	const double setup_time = sw.lap();

	fs::create_directories(dir / "meshes");
	std::ofstream csv(dir / (tag + ".csv"));
	write_timing_header(csv);

	Eigen::MatrixXd sol;
	double algorithm_cum = setup_time;
	std::cout << tag << " setup: " << setup_time << " s" << std::endl;

	for (int it = 0; it <= stop.uniform_levels; ++it)
	{
		IterTimes t;
		Stopwatch lap;

		sol.resize(0, 0); // nonlinear solves start from sol, which is still sized for the previous mesh
		state.solve(sol);
		disable_fp_traps();
		t.solve = lap.lap();

		io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		if (stats.num_dofs == 0) // the nonlinear forms do not set it
			stats.num_dofs = sol.rows();
		t.compute_errors = lap.lap();

		const auto &pf = state.variational_formulation->output_timings();
		const int n_el = mesh_ptr->n_elements();

		if (opt.dump_meshes)
		{
			lap.lap();
			dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_it" + std::to_string(it) + ".vtu"));
			t.io = lap.lap();
		}

		double eta_l2 = std::numeric_limits<double>::quiet_NaN(), eta_h1 = eta_l2;
		json fine_info;
		if (use_estimator)
		{
			lap.lap();
			const EstimatorResult e = estimate_by_uniform_refinement_both(state, sol, fine_args(args, opt), fine);
			eta_l2 = std::sqrt(e.l2.sum());
			eta_h1 = std::sqrt(e.h1_semi.sum());
			fine_info = e.fine_solver_info;
			t.conv_check = lap.lap(); // not counted
		}
		const double err = stop_error(opt, has_ref, stats, eta_l2, eta_h1);

		std::cout << tag << " it " << it << ": n_el " << n_el << ", dofs " << stats.num_dofs
				  << ", " << err_label(opt, has_ref) << " " << err << std::endl;

		if (it == 0)
			result.first_err = err;
		result.last_err = err;

		std::string stop_reason = check_stop(uniform_stop, err, result.first_err, stats.num_dofs);
		if (stop_reason.empty() && it == stop.uniform_levels)
			stop_reason = "Reached uniform level " + std::to_string(stop.uniform_levels);

		if (stop_reason.empty())
		{
			lap.lap();
			state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
				if (non_conforming)
				{
					mm.refine(1, 0.5);
					mm.prepare_mesh();
				}
				else
					refine_standard_mesh(mm, 1, args);
			});
			t.refine = lap.lap();
		}

		algorithm_cum += t.algorithm();
		write_timing_row(csv, method_name(method), p, "NA",
						 it, n_el, stats, t, algorithm_cum, pf, has_ref, eta_l2, eta_h1, fine_info);

		csv.flush();
		if (!stop_reason.empty())
		{
			std::cout << stop_reason << std::endl;
			final_reason = stop_reason;
			break;
		}
	}

	dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_final.vtu"));
	state.variational_formulation->export_data(sol);
	mark_completed(dir, tag, result.first_err, result.last_err, final_reason);
	return result;
}
