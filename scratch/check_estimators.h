#pragma once

#include "common.h"
#include "estimator.h"

#include <set>

// Needs an exact solution. Meshes are refined with the true error, so every estimator sees the same meshes.
inline void check_estimators(const fs::path &dir, json args, const int p, const Options &opt, const int rounds, const double theta = 0.5)
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
		const auto mk = mark_error_percentage(est, theta);
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
		disable_fp_traps();
		const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
		const Eigen::VectorXd err2 = state.variational_formulation->compute_element_errors(sol);
		const std::vector<int> truth = mark_error_percentage(err2, theta);
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
