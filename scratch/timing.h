#pragma once

#include "common.h"

#include <chrono>
#include <fstream>
#include <string>

using Clock = std::chrono::steady_clock;

// Returns seconds since construction or the previous lap().
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

// Wall-clock time spent in each part of one iteration (seconds).
struct IterTimes
{
	double solve = 0;          // state.solve(sol): everything PolyFEM does for the solve
	double compute_errors = 0; // compute_errors(sol): error against the reference solution
	double estimator = 0;      // compute_element_errors(sol): adaptive only
	double marking = 0;        // mark_bulk: adaptive only
	double refine = 0;         // update_mesh: refine (+ prepare_mesh)
	double io = 0;             // mesh .vtu dump: output only, excluded from the algorithm time

	// Time attributable to the method itself (no I/O).
	double algorithm() const { return solve + compute_errors + estimator + marking + refine; }
};

inline void write_timing_header(std::ofstream &csv)
{
	csv << "method,p,theta,iter,n_elements,n_dofs,h,l2,h1_semi,"
		<< "w_solve,w_compute_errors,w_estimator,w_marking,w_refine,w_io,w_algorithm,w_algorithm_cum,"
		<< "pf_loading_mesh,pf_building_basis,pf_poly_basis,pf_assemble_stiffness,"
		<< "pf_assemble_mass,pf_assign_rhs,pf_solving,pf_total\n";
}

// n_elements / stats describe the mesh that was SOLVED in this iteration.
inline void write_timing_row(std::ofstream &csv, const std::string &method, const int p,
							 const std::string &theta, const int it, const int n_elements,
							 const io::OutStatsData &stats, const IterTimes &t,
							 const double algorithm_cum, const io::OutRuntimeData &pf)
{
	csv << method << "," << p << "," << theta << "," << it << ","
		<< n_elements << "," << stats.num_dofs << "," << stats.mesh_size << ","
		<< stats.l2_err << "," << stats.h1_semi_err << ","
		<< t.solve << "," << t.compute_errors << "," << t.estimator << ","
		<< t.marking << "," << t.refine << "," << t.io << ","
		<< t.algorithm() << "," << algorithm_cum << ","
		<< pf.loading_mesh_time << "," << pf.building_basis_time << ","
		<< pf.computing_poly_basis_time << "," << pf.assembling_stiffness_mat_time << ","
		<< pf.assembling_mass_mat_time << "," << pf.assigning_rhs_time << ","
		<< pf.solving_time << "," << pf.total_time()
		<< std::endl; // flush each row
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
		<< tm.solving_time << "," << tm.total_time() << std::endl; // flush each row
}