#pragma once

#include "common.h"

#include <polyfem/utils/par_for.hpp>

#include <chrono>
#include <fstream>
#include <string>

using Clock = std::chrono::steady_clock;

struct Stopwatch
{
	Clock::time_point t0 = Clock::now();
	double lap()
	{
		const auto t1 = Clock::now();
		const double s = std::chrono::duration<double>(t1 - t0).count();
		t0 = t1;
		return s;
	}
};

struct IterTimes
{
	double solve = 0;
	double compute_errors = 0;
	double estimator = 0;
	double marking = 0;
	double refine = 0;
	double io = 0;             // excluded from the algorithm time
	double conv_check = 0;     // estimator only used to decide to stop, excluded from the algorithm time

	double algorithm() const { return solve + compute_errors + estimator + marking + refine; }
};

inline void write_timing_header(std::ofstream &csv)
{
	csv << "method,p,theta,iter,n_elements,n_dofs,h,l2,h1_semi,"
		<< "w_solve,w_compute_errors,w_estimator,w_marking,w_refine,w_io,w_algorithm,w_algorithm_cum,"
		<< "pf_loading_mesh,pf_building_basis,pf_poly_basis,pf_assemble_stiffness,"
		<< "pf_assemble_mass,pf_assign_rhs,pf_solving,pf_total,eta_l2,w_conv_check,eta_h1,n_threads,"
		<< "lin_iters,lin_res,fine_lin_iters,fine_lin_res,"
		<< "fine_t_factorize,fine_t_setup,fine_t_iterate,fine_t_solve_total,"
		<< "te_mesh,te_fine_solve,te_prolong,te_integrate,tf_basis,tf_assemble,tf_rhs,tf_solving,"
		<< "newton_iters,newton_subsolves,fine_newton_iters,fine_newton_subsolves\n";
}

// has_reference = false: l2 / h1_semi are written as nan (PolyFEM reports the norm of u_h there, not an error)
inline void write_timing_row(std::ofstream &csv, const std::string &method, const int p,
							 const std::string &theta, const int it, const int n_elements,
							 const io::OutStatsData &stats, const IterTimes &t,
							 const double algorithm_cum, const io::OutRuntimeData &pf,
							 const bool has_reference = true,
							 const double eta_l2 = std::numeric_limits<double>::quiet_NaN(),
							 const double eta_h1 = std::numeric_limits<double>::quiet_NaN(),
							 const json &fine_solver_info = json())
{
	const double nan = std::numeric_limits<double>::quiet_NaN();
	// iterative solvers (Hypre: num_iterations, final_res_norm = final relative residual); nan for direct ones
	const auto num = [&](const json &j, const char *key) { return j.is_object() && j.contains(key) && j[key].is_number() ? j[key].get<double>() : nan; };
	// Hypre reports num_iterations / final_res_norm, Eigen iterative solvers (e.g. CG) solver_iter / solver_error
	const auto num2 = [&](const json &j, const char *a, const char *b) { const double v = num(j, a); return std::isnan(v) ? num(j, b) : v; };
	csv << method << "," << p << "," << theta << "," << it << ","
		<< n_elements << "," << stats.num_dofs << "," << stats.mesh_size << ","
		<< (has_reference ? stats.l2_err : nan) << "," << (has_reference ? stats.h1_semi_err : nan) << ","
		<< t.solve << "," << t.compute_errors << "," << t.estimator << ","
		<< t.marking << "," << t.refine << "," << t.io << ","
		<< t.algorithm() << "," << algorithm_cum << ","
		<< pf.loading_mesh_time << "," << pf.building_basis_time << ","
		<< pf.computing_poly_basis_time << "," << pf.assembling_stiffness_mat_time << ","
		<< pf.assembling_mass_mat_time << "," << pf.assigning_rhs_time << ","
		<< pf.solving_time << "," << pf.total_time() << "," << eta_l2 << "," << t.conv_check << "," << eta_h1 << ","
		<< polyfem::utils::NThread::get().num_threads() << ","
		<< num2(stats.solver_info, "num_iterations", "solver_iter") << "," << num2(stats.solver_info, "final_res_norm", "solver_error") << ","
		<< num2(fine_solver_info, "num_iterations", "solver_iter") << "," << num2(fine_solver_info, "final_res_norm", "solver_error");
	for (const char *k : {"t_factorize", "t_setup", "t_iterate", "t_solve_total", "te_mesh", "te_fine_solve", "te_prolong",
						  "te_integrate", "tf_basis", "tf_assemble", "tf_rhs", "tf_solving"})
		csv << "," << num(fine_solver_info, k);
	// nonlinear solves: solver_info is a list of subsolves (load steps / AL), each {"info": {"iterations": n}}
	const auto newton = [&](const json &j, const bool count) {
		const json &list = j.is_object() && j.contains("subsolves") ? j["subsolves"] : j;
		if (!list.is_array())
			return nan;
		double it = 0;
		for (const auto &s : list)
			if (s.contains("info") && s["info"].contains("iterations"))
				it += s["info"]["iterations"].get<double>();
		return count ? double(list.size()) : it;
	};
	csv << "," << newton(stats.solver_info, false) << "," << newton(stats.solver_info, true)
		<< "," << newton(fine_solver_info, false) << "," << newton(fine_solver_info, true);
	csv << std::endl;
}

// Older CSV format used by the convergence runs (per-method, with PolyFEM's own timings).
inline void write_header(std::ofstream &csv)
{
	csv << "method,p,iter,n_elements,n_dofs,h,l2,h1_semi,t_solve,t_total\n";
}

inline void write_row(std::ofstream &csv, const Method method, const int p, const int it,
					  const mesh::Mesh &mesh, const io::OutStatsData &stats, const io::OutRuntimeData &tm)
{
	csv << method_name(method) << "," << p << "," << it << ","
		<< mesh.n_elements() << "," << stats.num_dofs << "," << stats.mesh_size << ","
		<< stats.l2_err << "," << stats.h1_semi_err << ","
		<< tm.solving_time << "," << tm.total_time() << std::endl;
}