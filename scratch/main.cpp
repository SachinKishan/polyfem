
/*
#include "common.h"
#include "timing.h"
#include "checks.h" // not used below; included so the angle / convergence tests can be called from main

void error_estimation_simple()
{
    // uniformly refine once
    // solve problem
    // compare the passed solution with the fine solution at the coarse points
    // if it's above some metric, refine/coarsen





}

void test_adaptive_refinement(const fs::path &dir, json args, const int p, const Options &opt, float theta = 0.5)
{
	const std::string tag = method_name(Method::NCNonUniform) + "_p" + std::to_string(p) + "_adaptive_theta_" + std::to_string(theta);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);

	const int max_iter = 100;
	const long long max_dofs = 100000;
	const double absolute_tol = 1e-4;
	double relative_tol = 1e-4;

	// Setup (mesh load + initial refinement + set_mesh) is timed separately.
	Stopwatch sw;
	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, args);
	visit_nc_mesh(*m, [&](auto &nc) { nc.refine(opt.base_refs, 0.5); nc.prepare_mesh(); });

	// Non-owning handle. Stays valid because the formulation owns the same object after
	// set_mesh, and update_mesh edits it in place without replacing it.
	mesh::Mesh *mesh_ptr = m.get();
	state.variational_formulation->set_mesh(std::move(m));
	const double setup_time = sw.lap();

	fs::create_directories(dir / "meshes");
	std::ofstream csv(dir / (tag + ".csv"));
	write_timing_header(csv);

	Eigen::MatrixXd sol;
	double algorithm_cum = setup_time; // setup counts toward the algorithm total
	std::cout << tag << " setup: " << setup_time << " s" << std::endl;

	for (int it = 0; it < max_iter; ++it)
	{
		IterTimes t;
		Stopwatch lap;

		state.solve(sol);
		t.solve = lap.lap();

		const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		t.compute_errors = lap.lap();

		const auto &pf = state.variational_formulation->output_timings();
		const int n_el = mesh_ptr->n_elements(); // mesh that was just solved

		std::cout << tag << " it " << it << ": n_el " << n_el
				  << ", dofs " << stats.num_dofs << ", L2 " << stats.l2_err << std::endl;

		if (opt.dump_meshes)
		{
			lap.lap();
			dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_it" + std::to_string(it) + ".vtu"));
			t.io = lap.lap();
		}

		if (it == 0)
			relative_tol = relative_tol * stats.l2_err;

		std::string stop_reason;
		if (stats.num_dofs > max_dofs)
			stop_reason = "Reached max dofs";
		else if (it + 1 == max_iter)
			stop_reason = "Failed to converge";
		else if (stats.l2_err < relative_tol)
			stop_reason = "Converged with relative error criterion";
		else if (stats.l2_err < absolute_tol)
			stop_reason = "Converged with absolute error criterion";

		if (stop_reason.empty())
		{
			lap.lap();
			const Eigen::VectorXd eta2 = state.variational_formulation->compute_element_errors(sol);
			t.estimator = lap.lap();

			const auto marked = mark_bulk(eta2, theta);
			t.marking = lap.lap();

			std::cout << "eta2.size=" << eta2.size() << " n_elements=" << n_el
					  << " sum(eta2)=" << eta2.sum()
					  << " l2^2=" << stats.l2_err * stats.l2_err
					  << " h1^2=" << stats.h1_semi_err * stats.h1_semi_err
					  << " marked=" << marked.size() << std::endl;
			lap.lap(); // do not charge the printing above to refinement

			state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
				visit_nc_mesh(mm, [&](auto &nc) { nc.refine_elements(marked); nc.prepare_mesh(); });
			});
			t.refine = lap.lap();
		}

		algorithm_cum += t.algorithm();
		write_timing_row(csv, method_name(Method::NCNonUniform), p, std::to_string(theta),
						 it, n_el, stats, t, algorithm_cum, pf);

		if (!stop_reason.empty())
		{
			std::cout << stop_reason << std::endl;
			break;
		}
	}

	dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_final.vtu"));
	state.variational_formulation->export_data(sol); // final solution, same paraview settings as uniform
}

void test_uniform_refinement(const fs::path &dir, json args, const int p, const Options &opt)
{
	const std::string tag = method_name(Method::NCUniform) + "_p" + std::to_string(p);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);

	// Keep these identical to the adaptive run.
	const int max_iter = 100;
	const long long max_dofs = 100000;
	const double absolute_tol = 1e-4;
	double relative_tol = 1e-4;

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

	for (int it = 0; it < max_iter; ++it)
	{
		IterTimes t;
		Stopwatch lap;

		state.solve(sol);
		t.solve = lap.lap();

		const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		t.compute_errors = lap.lap();

		const auto &pf = state.variational_formulation->output_timings();
		const int n_el = mesh_ptr->n_elements();

		std::cout << tag << " it " << it << ": n_el " << n_el
				  << ", dofs " << stats.num_dofs << ", L2 " << stats.l2_err << std::endl;

		if (opt.dump_meshes)
		{
			lap.lap();
			dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_it" + std::to_string(it) + ".vtu"));
			t.io = lap.lap();
		}

		if (it == 0)
			relative_tol = relative_tol * stats.l2_err;

		std::string stop_reason;
		if (stats.num_dofs > max_dofs)
			stop_reason = "Reached max dofs";
		else if (it + 1 == max_iter)
			stop_reason = "Failed to converge";
		else if (stats.l2_err < relative_tol)
			stop_reason = "Converged with relative error criterion";
		else if (stats.l2_err < absolute_tol)
			stop_reason = "Converged with absolute error criterion";

		if (stop_reason.empty())
		{
			lap.lap();
			state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
				mm.refine(1, 0.5);
				mm.prepare_mesh();
			});
			t.refine = lap.lap();
		}

		algorithm_cum += t.algorithm();
		write_timing_row(csv, method_name(Method::NCUniform), p, "NA",
						 it, n_el, stats, t, algorithm_cum, pf);

		if (!stop_reason.empty())
		{
			std::cout << stop_reason << std::endl;
			break;
		}
	}

	dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_final.vtu"));
	state.variational_formulation->export_data(sol);
}

int main()
{
	// ====================== edit freely ======================
	const std::string working_dir = "/Users/sak9889/WORK/Projects/BETTER_ITR/WorkingDirectory";

	Options opt;
	opt.base_refs = 0;
	opt.nonuniform_passes = 1;
	opt.plane = {/*axis=0, /*position=0.5, /*use_barycenter=true};
	opt.dump_meshes = true;
	// =========================================================

	fs::create_directories(working_dir);

	const std::string mesh = "//Users/sak9889/WORK/Projects/BETTER_ITR/meshes/square.obj";

	const json args = make_args(mesh);
	const fs::path dir = mesh_output_dir(working_dir, mesh); // <working_dir>/square/
	std::ofstream(dir / "args.json") << args.dump(2) << "\n";

	for (auto theta : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
		test_adaptive_refinement(dir, args, 2, opt, theta);

	test_uniform_refinement(dir, args, 2, opt);

	return EXIT_SUCCESS;
}
*/

#include "common.h"
#include "timing.h"
#include "checks.h" // not used below; included so the angle / convergence tests can be called from main
#include "estimator.h"

#include <set>

// use_fine_estimator = false: PolyFEM's compute_element_errors
// use_fine_estimator = true : coarse-vs-fine estimator from estimator.h
void test_adaptive_refinement(const fs::path &dir, json args, const int p, const Options &opt, float theta = 0.5,
							  const bool use_fine_estimator = false)
{
	const std::string tag = method_name(Method::NCNonUniform) + "_p" + std::to_string(p)
							+ (use_fine_estimator ? "_fine" : "") + "_adaptive_theta_" + std::to_string(theta);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);
	FineSolver fine; // initialised on first use, only when use_fine_estimator

	const int max_iter = 100;
	const long long max_dofs = 100000;
	const double absolute_tol = 1e-4;
	double relative_tol = 1e-4;

	// Setup (mesh load + initial refinement + set_mesh) is timed separately.
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

	for (int it = 0; it < max_iter; ++it)
	{
		IterTimes t;
		Stopwatch lap;

		state.solve(sol);
		t.solve = lap.lap();

		const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		t.compute_errors = lap.lap();

		const auto &pf = state.variational_formulation->output_timings();
		const int n_el = mesh_ptr->n_elements(); // mesh that was just solved

		std::cout << tag << " it " << it << ": n_el " << n_el
				  << ", dofs " << stats.num_dofs << ", L2 " << stats.l2_err << std::endl;

		if (opt.dump_meshes)
		{
			lap.lap();
			dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_it" + std::to_string(it) + ".vtu"));
			t.io = lap.lap();
		}

		if (it == 0)
			relative_tol = relative_tol * stats.l2_err;

		std::string stop_reason;
		if (stats.num_dofs > max_dofs)
			stop_reason = "Reached max dofs";
		else if (it + 1 == max_iter)
			stop_reason = "Failed to converge";
		else if (stats.l2_err < relative_tol)
			stop_reason = "Converged with relative error criterion";
		else if (stats.l2_err < absolute_tol)
			stop_reason = "Converged with absolute error criterion";

		if (stop_reason.empty())
		{
			lap.lap();
			const Eigen::VectorXd eta2 = use_fine_estimator
				? estimate_by_uniform_refinement(state, sol, args, fine, EstimatorNorm::L2)
				: state.variational_formulation->compute_element_errors(sol);
			t.estimator = lap.lap();



			const auto marked = mark_bulk(eta2, theta);
			t.marking = lap.lap();

			std::cout << "eta2.size=" << eta2.size() << " n_elements=" << n_el
					  << " sum(eta2)=" << eta2.sum()
					  << " l2^2=" << stats.l2_err * stats.l2_err
					  << " h1^2=" << stats.h1_semi_err * stats.h1_semi_err
					  << " marked=" << marked.size() << std::endl;
			lap.lap(); // do not charge the printing above to refinement

			state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
				visit_nc_mesh(mm, [&](auto &nc) { nc.refine_elements(marked); nc.prepare_mesh(); });
			});
			t.refine = lap.lap();
		}

		algorithm_cum += t.algorithm();
		write_timing_row(csv, method_name(Method::NCNonUniform), p, std::to_string(theta),
						 it, n_el, stats, t, algorithm_cum, pf);

		if (!stop_reason.empty())
		{
			std::cout << stop_reason << std::endl;
			break;
		}
	}

	dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_final.vtu"));
	state.variational_formulation->export_data(sol); // final solution, same paraview settings as uniform
}

void test_uniform_refinement(const fs::path &dir, json args, const int p, const Options &opt)
{
	const std::string tag = method_name(Method::NCUniform) + "_p" + std::to_string(p);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);

	// Keep these identical to the adaptive run.
	const int max_iter = 100;
	const long long max_dofs = 100000;
	const double absolute_tol = 1e-4;
	double relative_tol = 1e-4;

	Stopwatch sw;
	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, args);
	visit_nc_mesh(*m, [&](auto &nc) { nc.refine(opt.base_refs, 0.5); nc.prepare_mesh(); });

	mesh::Mesh *mesh_ptr = m.get(); // non-owning, valid while the formulation owns it
	state.variational_formulation->set_mesh(std::move(m));
	const double setup_time = sw.lap();

	fs::create_directories(dir / "meshes");
	std::ofstream csv(dir / (tag + ".csv"));
	write_timing_header(csv);

	Eigen::MatrixXd sol;
	double algorithm_cum = setup_time;
	std::cout << tag << " setup: " << setup_time << " s" << std::endl;

	for (int it = 0; it < max_iter; ++it)
	{
		IterTimes t;
		Stopwatch lap;

		state.solve(sol);
		t.solve = lap.lap();

		const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		t.compute_errors = lap.lap();

		const auto &pf = state.variational_formulation->output_timings();
		const int n_el = mesh_ptr->n_elements();

		std::cout << tag << " it " << it << ": n_el " << n_el
				  << ", dofs " << stats.num_dofs << ", L2 " << stats.l2_err << std::endl;

		if (opt.dump_meshes)
		{
			lap.lap();
			dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_it" + std::to_string(it) + ".vtu"));
			t.io = lap.lap();
		}

		if (it == 0)
			relative_tol = relative_tol * stats.l2_err;

		std::string stop_reason;
		if (stats.num_dofs > max_dofs)
			stop_reason = "Reached max dofs";
		else if (it + 1 == max_iter)
			stop_reason = "Failed to converge";
		else if (stats.l2_err < relative_tol)
			stop_reason = "Converged with relative error criterion";
		else if (stats.l2_err < absolute_tol)
			stop_reason = "Converged with absolute error criterion";

		if (stop_reason.empty())
		{
			lap.lap();
			state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
				mm.refine(1, 0.5);
				mm.prepare_mesh();
			});
			t.refine = lap.lap();
		}

		algorithm_cum += t.algorithm();
		write_timing_row(csv, method_name(Method::NCUniform), p, "NA",
						 it, n_el, stats, t, algorithm_cum, pf);

		if (!stop_reason.empty())
		{
			std::cout << stop_reason << std::endl;
			break;
		}
	}

	dump_mesh_vtu(*mesh_ptr, dir / "meshes" / (tag + "_final.vtu"));
	state.variational_formulation->export_data(sol);
}

// Scores the estimators against the true per-element error (L2^2 from compute_element_errors) on
// the same meshes. The meshes are refined with the true error (theta = 0.5), so every estimator sees
// the same non-uniform NC meshes. Needs a problem with an exact solution.
//   effectivity = sqrt(sum eta^2 / sum err^2)   (1 is ideal; a constant factor is harmless for marking)
//   corr        = correlation of log10 eta^2 and log10 err^2 over the elements
//   overlap     = share of the true Doerfler set (theta) that the estimator also marks
void check_estimators(const fs::path &dir, json args, const int p, const Options &opt, const int rounds, const double theta = 0.5)
{
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, "estimator_check_p" + std::to_string(p));

	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, args);
	visit_nc_mesh(*m, [&](auto &nc) { nc.refine(opt.base_refs, 0.5); nc.prepare_mesh(); });
	state.variational_formulation->set_mesh(std::move(m));
	FineSolver fine;

	const auto log_corr = [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
		const Eigen::ArrayXd x = a.array().max(1e-300).log10(), y = b.array().max(1e-300).log10();
		const Eigen::ArrayXd dx = x - x.mean(), dy = y - y.mean();
		return (dx * dy).sum() / std::sqrt((dx * dx).sum() * (dy * dy).sum());
	};
	const auto overlap = [&](const Eigen::VectorXd &est, const std::vector<int> &truth) {
		const auto mk = mark_bulk(est, theta);
		const std::set<int> s(mk.begin(), mk.end());
		int hit = 0;
		for (int e : truth)
			hit += s.count(e);
		return std::make_pair(double(hit) / truth.size(), int(mk.size()));
	};

	std::ofstream csv(dir / ("estimator_check_p" + std::to_string(p) + ".csv"));
	csv << "round,n_elements,estimator,effectivity,corr,overlap,n_marked,n_marked_true\n";
	std::cout << std::setprecision(3);
	for (int r = 0; r < rounds; ++r)
	{
		Eigen::MatrixXd sol;
		state.solve(sol);
		const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		const Eigen::VectorXd err2 = state.variational_formulation->compute_element_errors(sol);
		const std::vector<int> truth = mark_bulk(err2, theta);
		const int n_el = err2.size();

		std::vector<std::pair<std::string, Eigen::VectorXd>> est;
		try
		{
			est.emplace_back("nodal (old)", error_estimation_simple(state, sol, args, fine));
		}
		catch (const std::exception &e)
		{
			std::cout << "nodal (old) estimator threw: " << e.what() << std::endl;
		}
		est.emplace_back("integrated L2", estimate_by_uniform_refinement(state, sol, args, fine, EstimatorNorm::L2));
		est.emplace_back("integrated H1semi", estimate_by_uniform_refinement(state, sol, args, fine, EstimatorNorm::H1Semi));

		std::cout << "== round " << r << ": " << n_el << " elements, true L2 " << stats.l2_err
				  << ", true H1semi " << stats.h1_semi_err << ", true Doerfler set " << truth.size() << " elements" << std::endl;
		for (const auto &[name, eta2] : est)
		{
			// the H1 estimate is compared with the global H1 error; per element only L2 is available
			const double ref2 = name == "integrated H1semi" ? stats.h1_semi_err * stats.h1_semi_err : err2.sum();
			const double eff = std::sqrt(eta2.sum() / ref2);
			const double corr = log_corr(eta2, err2);
			const auto [ov, nm] = overlap(eta2, truth);
			std::cout << "   " << std::left << std::setw(18) << name << " effectivity " << std::setw(8) << eff
					  << " corr " << std::setw(8) << corr << " overlap " << std::setw(6) << ov << " marked " << nm << std::endl;
			csv << r << "," << n_el << "," << name << "," << eff << "," << corr << "," << ov << "," << nm << "," << truth.size() << "\n";
		}

		state.variational_formulation->update_mesh([&](mesh::Mesh &mm) {
			visit_nc_mesh(mm, [&](auto &nc) { nc.refine_elements(truth); nc.prepare_mesh(); });
		});
	}
}

int main(int argc, char **argv)
{
	// ====================== edit freely ======================
	// optional argv[1] overrides the output folder
	const std::string working_dir = argc > 1 ? argv[1] : "/Users/sak9889/WORK/Projects/BETTER_ITR/WorkingDirectory";

	Options opt;
	opt.base_refs = 0;
	opt.nonuniform_passes = 1;
	opt.plane = {/*axis=*/0, /*position=*/0.5, /*use_barycenter=*/true};
	opt.dump_meshes = true;
	// =========================================================

	fs::create_directories(working_dir);

	const std::string mesh = "//Users/sak9889/WORK/Projects/BETTER_ITR/meshes/lshape.obj";

	//const json args = make_args(mesh);

	const std::string args_file = "//Users/sak9889/WORK/Projects/BETTER_ITR/inputs/lshape.json";
    const json args = load_args(args_file);


	const fs::path dir = mesh_output_dir(working_dir, mesh); // <working_dir>/square/
	std::ofstream(dir / "args.json") << args.dump(2) << "\n";

	// optional argv[2]: "check-estimator" scores the estimators against the true error,
	// "adaptive-fine" runs the adaptive loop with the uniform-refinement estimator (no exact solution used for marking)
	const std::string mode = argc > 2 ? argv[2] : "";
	if (mode == "check-estimator")
	{
		for (int p : {1, 2})
			check_estimators(dir, args, p, opt, /*rounds=*/6);
		return EXIT_SUCCESS;
	}
	if (mode == "adaptive-fine")
	{
		test_adaptive_refinement(dir, args, 2, opt, 0.5, true);
		return EXIT_SUCCESS;
	}

	for (auto theta : {0.5})
		test_adaptive_refinement(dir, args, 2, opt, theta, false);

	//test_uniform_refinement(dir, args, 2, opt);

	return EXIT_SUCCESS;
}