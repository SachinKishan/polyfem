#pragma once

#include "common.h"
#include "timing.h"
#include "estimator.h"
#include "stopping.h"
#include "resume.h"

// Uses StopCriteria and Options from main.cpp, so it is included after them.

inline void test_adaptive_refinement(const fs::path &dir, json args, const int p, const Options &opt, const StopCriteria &stop,
									 float theta = 0.5, const bool use_fine_estimator = false)
{
	const std::string tag = method_name(Method::NCNonUniform) + "_p" + std::to_string(p)
							+ (use_fine_estimator ? "_fine" : "") + "_adaptive_theta_" + std::to_string(theta);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);
	FineSolver fine;
	fine.warm_start = opt.fine_warm_start;

	const bool has_ref = opt.use_exact_error && has_reference(args);
	if (!has_ref && !use_fine_estimator)
		log_and_throw_error("{}: no output.reference in the args, use the fine estimator (use_fine_estimator = true).", tag);

	if (opt.reuse_finished_runs && read_completed(dir).contains(tag))
	{
		std::cout << tag << ": already finished in " << dir << ", skipped (--fresh redoes it)" << std::endl;
		return;
	}

	double err0 = std::numeric_limits<double>::quiet_NaN();
	double last_err = std::numeric_limits<double>::quiet_NaN();
	std::string final_reason;

	Stopwatch sw;
	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, args);
	visit_nc_mesh(*m, [&](auto &nc) { nc.refine(opt.base_refs, 0.5); nc.prepare_mesh(); });

	mesh::Mesh *mesh_ptr = m.get();
	state.variational_formulation->set_mesh(std::move(m));
	const double setup_time = sw.lap();

	fs::create_directories(dir / "meshes");
	std::ofstream csv(dir / (tag + ".csv"));
	write_timing_header(csv);

	Eigen::MatrixXd sol;
	double algorithm_cum = setup_time;
	std::cout << tag << " setup: " << setup_time << " s" << std::endl;

	for (int it = 0; it < stop.max_iter; ++it)
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

		// PolyFEM's compute_element_errors is L2 only
		json fine_info;
		const auto estimate = [&]() {
			lap.lap();
			EstimatorResult e;
			if (use_fine_estimator)
			{
				e = estimate_by_uniform_refinement_both(state, sol, fine_args(args, opt), fine);
				fine_info = e.fine_solver_info;
			}
			else
				e.l2 = e.h1_semi = state.variational_formulation->compute_element_errors(sol);
			t.estimator = lap.lap();
			return e;
		};

		Eigen::VectorXd eta2;
		double eta_l2 = std::numeric_limits<double>::quiet_NaN(), eta_h1 = eta_l2;
		if (!has_ref)
		{
			const EstimatorResult e = estimate();
			eta_l2 = std::sqrt(e.l2.sum());
			eta_h1 = std::sqrt(e.h1_semi.sum());
			eta2 = e[opt.mark_norm];
		}
		const double err = stop_error(opt, has_ref, stats, eta_l2, eta_h1);

		std::cout << tag << " it " << it << ": n_el " << n_el << ", dofs " << stats.num_dofs
				  << ", " << err_label(opt, has_ref) << " " << err << std::endl;

		if (it == 0)
			err0 = err;
		last_err = err;

		std::string stop_reason = check_stop(stop, err, err0, stats.num_dofs);
		if (stop_reason.empty() && it + 1 == stop.max_iter)
			stop_reason = "Failed to converge";

		// nothing gets marked with this estimate, so it is not counted as estimator time
		if (!has_ref && !stop_reason.empty())
		{
			t.conv_check = t.estimator;
			t.estimator = 0;
		}

		if (stop_reason.empty())
		{
			if (has_ref)
				eta2 = estimate()[opt.mark_norm];

			//marking according to the estimated error

			const auto marked = mark_elements(eta2, theta, opt.mark_scheme);



			t.marking = lap.lap();
			// the estimator solve is bigger still, so stop before refining past max_dofs, not after
			const double growth = 1 + (std::pow(2, mesh_ptr->dimension()) - 1) * double(marked.size()) / n_el;
			if (marked.empty())
				stop_reason = "No elements marked";
			else if (stats.num_dofs * growth > stop.max_dofs)
				stop_reason = "Reached max dofs";

			std::cout << "eta2.size=" << eta2.size() << " n_elements=" << n_el
					  << " sum(eta2)=" << eta2.sum()
					  << " l2^2=" << stats.l2_err * stats.l2_err
					  << " h1^2=" << stats.h1_semi_err * stats.h1_semi_err
					  << " marked=" << marked.size() << std::endl;
			lap.lap(); // do not charge the printing above to refinement

			if (stop_reason.empty())
			{
				state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
					visit_nc_mesh(mm, [&](auto &nc) { nc.refine_elements(marked); nc.prepare_mesh(); });
				});
				t.refine = lap.lap();
			}
		}

		algorithm_cum += t.algorithm();
		write_timing_row(csv, method_name(Method::NCNonUniform), p, std::to_string(theta),
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
	mark_completed(dir, tag, err0, last_err, final_reason);
}
