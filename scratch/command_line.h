#pragma once

#include "common.h"

#include <polyfem/utils/par_for.hpp>

#include <sstream>
#include <thread>

// Uses StopCriteria and Options from main.cpp, so it is included after them.

// Set before anything runs: MKL / OpenMP / Accelerate read these variables once, on first use
inline int set_threads(const int max_threads)
{
	const unsigned hw = std::thread::hardware_concurrency();
	const int n = max_threads <= 0 ? int(hw) : std::min(max_threads, int(hw));
	const std::string v = std::to_string(n);
	for (const char *var : {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"})
		setenv(var, v.c_str(), /*overwrite=*/1);
	utils::NThread::get().set_num_threads(n);
	std::cout << "threads: " << n << " (of " << hw << " cores)" << std::endl;
	return n;
}

// Positional: [output dir] [mode] [args file]. A tolerance flag switches to Tolerance mode, --levels is then only a cap.
inline void parse_command_line(const int argc, char **argv, StopCriteria &stop, Options &opt)
{
	bool levels = false, tol = false;
	std::vector<std::string> pos;
	for (int i = 1; i < argc; ++i)
	{
		const std::string a = argv[i];
		const auto value = [&]() -> std::string {
			if (i + 1 >= argc)
				log_and_throw_error("{} needs a value", a);
			return argv[++i];
		};
		if (a == "--levels")
		{
			stop.uniform_levels = std::stoi(value());
			levels = true;
		}
		else if (a == "--abs-tol")
		{
			stop.absolute_tol = std::stod(value());
			tol = true;
		}
		else if (a == "--rel-tol")
		{
			stop.relative_tol = std::stod(value());
			tol = true;
		}
		else if (a == "--uniform-abs-tol")
			stop.uniform_absolute_tol = std::stod(value());
		else if (a == "--uniform-rel-tol")
			stop.uniform_relative_tol = std::stod(value());
		else if (a == "--max-dofs")
			stop.max_dofs = std::stoll(value());
		else if (a == "--fresh")
			opt.reuse_finished_runs = false;
		else if (a == "--threads")
			opt.max_threads = std::stoi(value());
		else if (a == "--exact-error")
			opt.use_exact_error = true;
		else if (a == "--no-mesh-dumps")
			opt.dump_meshes = false;
		else if (a == "--thetas")
		{
			opt.thetas.clear();
			std::stringstream ss(value());
			for (std::string v; std::getline(ss, v, ',');)
				opt.thetas.push_back(std::stod(v));
		}
		else if (a == "--mark-norm")
		{
			const std::string v = value();
			if (v == "l2")
				opt.mark_norm = EstimatorNorm::L2;
			else if (v == "h1")
				opt.mark_norm = EstimatorNorm::H1Semi;
			else
				log_and_throw_error("--mark-norm takes l2 or h1, got {}", v);
		}
		else if (a == "--stop-norm")
		{
			const std::string v = value();
			if (v == "l2")
				opt.stop_norm = EstimatorNorm::L2;
			else if (v == "h1")
				opt.stop_norm = EstimatorNorm::H1Semi;
			else
				log_and_throw_error("--stop-norm takes l2 or h1, got {}", v);
		}
		else if (a == "--mark-scheme")
		{
			const std::string v = value();
			if (v == "error")
				opt.mark_scheme = MarkScheme::ErrorPercentage;
			else if (v == "element")
				opt.mark_scheme = MarkScheme::ElementPercentage;
			else
				log_and_throw_error("--mark-scheme takes error or element, got {}", v);
		}
		else if (a.rfind("--", 0) == 0)
			log_and_throw_error("unknown flag {} (known: --levels, --abs-tol, --rel-tol, --uniform-abs-tol, --uniform-rel-tol, --max-dofs, --fresh, --threads, --mark-norm, --stop-norm, --mark-scheme, --exact-error, --thetas, --no-mesh-dumps)", a);
		else
			pos.push_back(a);
	}
	if (tol)
		stop.mode = StopCriteria::Mode::Tolerance;
	else if (levels)
		stop.mode = StopCriteria::Mode::MatchUniform;

	if (pos.size() > 0)
		opt.working_dir = pos[0];
	if (pos.size() > 1)
		opt.mode = pos[1];
	if (pos.size() > 2)
		opt.args_file = pos[2];
}
