#pragma once

#include "common.h"
#include "resume.h"
#include "stopping.h"

#include <ctime>

// Uses StopCriteria and Options from main.cpp, so it is included after them.

inline void write_run_summary(const fs::path &run_dir, const int argc, char **argv, const StopCriteria &stop,
							  const Options &opt, const int n_threads, const int p, json args,
							  const UniformResult &uni, const StopCriteria &adaptive_stop)
{
	std::ofstream out(run_dir / "run_summary.txt");
	out << std::setprecision(10);

	const std::time_t now = std::time(nullptr);
	out << "Finished: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "\n";
	out << "Command:";
	for (int i = 0; i < argc; ++i)
		out << " " << std::quoted(argv[i]);
	out << "\nResults: " << fs::absolute(run_dir).string() << "\n";

	const json &geometry = first_geometry(args);
	out << "\n== Input ==\n"
		<< "Args file: " << opt.args_file << "\n"
		<< "Mesh: " << geometry.value("mesh", std::string("(not set)")) << "\n"
		<< "Mesh n_refs: " << geometry.value("n_refs", 0) << "\n"
		<< "Base refinements: " << opt.base_refs << "\n";

	out << "\n== Problem ==\n";
	if (args.contains("preset_problem"))
		out << "Preset problem: " << args["preset_problem"].dump() << "\n";
	else
		out << "Preset problem: none (boundary conditions from the args file)\n";
	out << "Materials: " << (args.contains("materials") ? args["materials"].dump() : "(not set)") << "\n"
		<< "Exact solution: " << (has_reference(args) ? "yes" : "no") << "\n"
		<< "Discretisation order: " << p << "\n";

	out << "\n== Stopping criteria ==\n"
		<< "Mode: " << mode_name(stop.mode) << "\n"
		<< "Uniform levels: " << stop.uniform_levels << "\n";
	if (stop.mode == StopCriteria::Mode::MatchUniform)
		out << "Uniform absolute tol: " << stop.uniform_absolute_tol << "\n"
			<< "Uniform relative tol: " << stop.uniform_relative_tol << "\n";
	out << "Uniform error: " << uni.first_err << " -> " << uni.last_err << "\n"
		<< "Adaptive absolute tol: " << adaptive_stop.absolute_tol << "\n"
		<< "Adaptive relative tol: " << adaptive_stop.relative_tol << "\n"
		<< "Max iterations: " << stop.max_iter << "\n"
		<< "Max dofs: " << stop.max_dofs << "\n";

	out << "\n== Options ==\n"
		<< "Mode: " << (opt.mode.empty() ? "default" : opt.mode) << "\n"
		<< "Threads: " << n_threads << "\n"
		<< "Marking norm: " << norm_name(opt.mark_norm) << "\n"
		<< "Stopping norm: " << norm_name(opt.stop_norm) << "\n"
		<< "Marking scheme: " << scheme_name(opt.mark_scheme) << "\n"
		<< "Error measure: " << (opt.use_exact_error && has_reference(args) ? "exact" : "estimator") << "\n"
		<< "Thetas:";
	for (const double theta : opt.thetas)
		out << " " << theta;
	out << "\nMesh dumps: " << (opt.dump_meshes ? "yes" : "no") << "\n"
		<< "Reuse finished runs: " << (opt.reuse_finished_runs ? "yes" : "no") << "\n";

	out << "\n== Runs ==\n";
	const json done = read_completed(run_dir);
	for (const auto &[tag, run] : done.items())
		out << tag << ": err " << json_err(run["first_err"]) << " -> " << json_err(run["last_err"])
			<< ", " << run["stop_reason"].get<std::string>() << "\n";

	// resolved_args.json has every value PolyFEM used, defaults included; a uniform run reused from
	// before it existed did not write one
	const fs::path resolved = run_dir / "resolved_args.json";
	if (fs::exists(resolved))
		out << "\n== PolyFEM settings (with defaults) ==\n" << std::ifstream(resolved).rdbuf() << "\n";
	else
		out << "\n== PolyFEM settings (args file only) ==\n" << args.dump(2) << "\n";
}
