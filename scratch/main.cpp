// Where things are (all in this folder):
//   run_uniform.h       uniform refinement loop, on the NC mesh or a standard mesh
//   run_adaptive.h      adaptive loop: solve, estimate, mark, refine
//   run_single.h        one solve, no refinement
//   stopping.h          stopping test, adaptive target, stop_criteria.json, results folder name
//   resume.h            completed.json: skipping runs that already finished
//   run_summary.h       run_summary.txt: every setting of a finished experiment
//   command_line.h      command line flags, thread setup
//   estimator.h         coarse-vs-fine error estimator, Doerfler marking
//   check_estimators.h  scores the estimators against the true error
//   checks.h            angle and convergence tests
//   timing.h            stopwatch, csv columns
//   common.h            mesh loading, args files, mesh dumps

#include "common.h"
#include "estimator.h"

// err is the exact error with --exact-error and an exact solution, else the fine estimator's eta, in opt.stop_norm (H1 semi by default).
// MatchUniform: the error uniform stops at is the adaptive target; absolute_tol / relative_tol are not used.
// Tolerance   : uniform and adaptive stop at absolute_tol / relative_tol; uniform_levels is only a cap.
struct StopCriteria
{
	enum class Mode
	{
		MatchUniform,
		Tolerance
	};
	Mode mode = Mode::MatchUniform;
	int uniform_levels = 7;
	double absolute_tol = 1e-5;
	double relative_tol = 1e-5;
	double uniform_absolute_tol = 1e-6; // MatchUniform only, 0 turns it off
	double uniform_relative_tol = 1e-6; // MatchUniform only, 0 turns it off
	int max_iter = 100;
	long long max_dofs = 2000000;
};

struct Options
{
	std::string working_dir;
	std::string args_file;
	std::string mode;
	int max_threads = 0; // 0 = all cores
	std::vector<double> thetas;
	EstimatorNorm mark_norm = EstimatorNorm::H1Semi;
	EstimatorNorm stop_norm = EstimatorNorm::H1Semi; // the error the stopping test, csvs and plots use
	MarkScheme mark_scheme = MarkScheme::ErrorPercentage;
	bool use_exact_error = false; // false: stopping and the csv use the estimator even when an exact solution exists
	bool reuse_finished_runs = true;
	bool dump_meshes = true;
	int base_refs = 1;
	int nonuniform_passes = 1; // convergence tests only
	Plane plane;
};

// these use the structs above
#include "checks.h" // not used below; included so the angle / convergence tests can be called from main
#include "check_estimators.h"
#include "command_line.h"
#include "resume.h"
#include "run_adaptive.h"
#include "run_single.h"
#include "run_summary.h"
#include "run_uniform.h"
#include "stopping.h"

int main(int argc, char **argv)
{
	// ================= defaults: edit freely, command-line flags override them =================
	StopCriteria stop;
	stop.mode = StopCriteria::Mode::MatchUniform;
	stop.uniform_levels = 4;
	stop.uniform_absolute_tol = 1e-6;
	stop.uniform_relative_tol = 1e-6;
	stop.absolute_tol = 1e-4;
	stop.relative_tol = 1e-4;
	stop.max_dofs = 2000000;

	Options opt;
	opt.working_dir = "/Users/sak9889/WORK/Projects/BETTER_ITR/WorkingDirectory";
	opt.args_file = "//Users/sak9889/WORK/Projects/BETTER_ITR/inputs/lshape_elastic.json"; // 2D linear elasticity, same L-shape
	//opt.args_file = "//Users/sak9889/WORK/Projects/BETTER_ITR/inputs/lshape.json"; // scalar Laplace
	//opt.args_file = "//Users/sak9889/WORK/Projects/BETTER_ITR/inputs/square_elastic_smooth.json"; // 2D linear elasticity, smooth sanity check
	opt.thetas = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	opt.max_threads = 0;
	opt.mark_norm = EstimatorNorm::H1Semi; // or EstimatorNorm::L2 (--mark-norm l2)
	opt.stop_norm = EstimatorNorm::H1Semi; // or EstimatorNorm::L2 (--stop-norm l2)
	opt.mark_scheme = MarkScheme::ErrorPercentage; // or MarkScheme::ElementPercentage
	opt.base_refs = 0;
	opt.nonuniform_passes = 1;
	opt.plane = {/*axis=*/0, /*position=*/0.5, /*use_barycenter=*/true};
	const int p = 2;
	// ===========================================================================================

	parse_command_line(argc, argv, stop, opt);
	const int n_threads = set_threads(opt.max_threads);

	fs::create_directories(opt.working_dir);
	json args = load_args(opt.args_file);
	args["solver"]["max_threads"] = n_threads;

	const fs::path dir = mesh_output_dir(opt.working_dir, opt.args_file);
	std::ofstream(dir / "args.json") << args.dump(2) << "\n";

	if (opt.mode == "single")
	{
		test_single_solve(dir, args, p, opt);
		return EXIT_SUCCESS;
	}

	const fs::path run_dir = dir / run_dir_name(stop, opt, n_threads);
	fs::create_directories(run_dir);
	std::ofstream(run_dir / "args.json") << args.dump(2) << "\n";
	std::cout << "results: " << run_dir << std::endl;

	if (opt.mode == "uniform-compare")
	{
		const UniformResult uni = test_uniform_refinement(run_dir, args, p, opt, stop, /*non_conforming=*/true);
		test_uniform_refinement(run_dir, args, p, opt, stop, /*non_conforming=*/false);
		const StopCriteria adaptive_stop = adaptive_criteria(stop, uni);
		write_stop_criteria(run_dir, stop, opt, uni, adaptive_stop);
		write_run_summary(run_dir, argc, argv, stop, opt, n_threads, p, args, uni, adaptive_stop);
		return EXIT_SUCCESS;
	}
	if (opt.mode == "uniform-standard")
	{
		// the standard mesh has no error without an exact solution: then there is nothing to derive
		const UniformResult uni = test_uniform_refinement(run_dir, args, p, opt, stop, /*non_conforming=*/false);
		const StopCriteria adaptive_stop = std::isfinite(uni.last_err) ? adaptive_criteria(stop, uni) : stop;
		write_stop_criteria(run_dir, stop, opt, uni, adaptive_stop);
		write_run_summary(run_dir, argc, argv, stop, opt, n_threads, p, args, uni, adaptive_stop);
		return EXIT_SUCCESS;
	}

	if (opt.mode == "check-estimator")
	{
		for (int order : {1, 2})
			check_estimators(dir, args, order, opt, /*rounds=*/6);
		return EXIT_SUCCESS;
	}

	if (opt.mode == "adaptive-fine")
	{
		test_adaptive_refinement(dir, args, p, opt, stop, 0.5, true);
		return EXIT_SUCCESS;
	}

	// uniform first: in MatchUniform mode its final error is the adaptive target
	const UniformResult uni = test_uniform_refinement(run_dir, args, p, opt, stop);
	const StopCriteria adaptive_stop = adaptive_criteria(stop, uni);
	print_stop_criteria(stop, opt, n_threads, uni, adaptive_stop);
	write_stop_criteria(run_dir, stop, opt, uni, adaptive_stop);

	for (const double theta : opt.thetas)
		test_adaptive_refinement(run_dir, args, p, opt, adaptive_stop, theta, true);

	test_single_solve(dir, args, p, opt);
	write_run_summary(run_dir, argc, argv, stop, opt, n_threads, p, args, uni, adaptive_stop);

	return EXIT_SUCCESS;
}
