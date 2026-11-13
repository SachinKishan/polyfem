#pragma once

#include "common.h"
#include "timing.h"

#include <polyfem/utils/par_for.hpp>

// Uses StopCriteria and Options from main.cpp, so it is included after them.

inline const char *mode_name(const StopCriteria::Mode mode)
{
	return mode == StopCriteria::Mode::MatchUniform ? "match-uniform" : "tolerance";
}

inline const char *norm_name(const EstimatorNorm norm)
{
	return norm == EstimatorNorm::L2 ? "l2" : "h1";
}

// the error the stopping test uses: exact error if asked for and available, else the fine estimator
inline double stop_error(const Options &opt, const bool has_ref, const io::OutStatsData &stats,
						 const double eta_l2, const double eta_h1)
{
	if (opt.stop_norm == EstimatorNorm::L2)
		return has_ref ? stats.l2_err : eta_l2;
	return has_ref ? stats.h1_semi_err : eta_h1;
}

inline std::string err_label(const Options &opt, const bool has_ref)
{
	return std::string(has_ref ? "" : "eta_") + (opt.stop_norm == EstimatorNorm::L2 ? "L2" : "H1");
}

inline const char *scheme_name(const MarkScheme scheme)
{
	return scheme == MarkScheme::ErrorPercentage ? "error" : "element";
}

struct UniformResult
{
	double first_err = std::numeric_limits<double>::quiet_NaN();
	double last_err = std::numeric_limits<double>::quiet_NaN();
};

// "" while the run continues
inline std::string check_stop(const StopCriteria &stop, const double err, const double err0, const long long dofs)
{
	if (dofs > stop.max_dofs)
		return "Reached max dofs";
	// small slack: theta = 1 refines like uniform and reaches the same error up to round-off,
	// which on the large solves is more than 1e-9 relative
	constexpr double slack = 1 + 1e-6;
	if (err <= stop.absolute_tol * slack)
		return "Converged with absolute error criterion";
	if (err <= stop.relative_tol * err0 * slack)
		return "Converged with relative error criterion";
	return "";
}

// levels_<N>, or tol_abs<A>_rel<R>_levels<N> in tolerance mode
inline std::string criteria_dir_name(const StopCriteria &stop)
{
	if (stop.mode == StopCriteria::Mode::MatchUniform)
		return "levels_" + std::to_string(stop.uniform_levels);
	char buf[128];
	std::snprintf(buf, sizeof(buf), "tol_abs%g_rel%g_levels%d", stop.absolute_tol, stop.relative_tol, stop.uniform_levels);
	return buf;
}

inline StopCriteria adaptive_criteria(const StopCriteria &stop, const UniformResult &uni)
{
	StopCriteria adaptive_stop = stop;
	if (stop.mode == StopCriteria::Mode::MatchUniform)
	{
		if (!std::isfinite(uni.last_err) || !std::isfinite(uni.first_err))
			log_and_throw_error("match-uniform: the uniform run has no error measure, nothing to match");
		adaptive_stop.absolute_tol = uni.last_err;
		adaptive_stop.relative_tol = uni.last_err / uni.first_err;
	}
	return adaptive_stop;
}

inline void write_stop_criteria(const fs::path &run_dir, const StopCriteria &stop, const Options &opt,
								const UniformResult &uni, const StopCriteria &adaptive_stop)
{
	std::ofstream(run_dir / "stop_criteria.json") << json{
		{"mode", mode_name(stop.mode)},
		{"threads", utils::NThread::get().num_threads()},
		{"mark_norm", norm_name(opt.mark_norm)},
		{"stop_norm", norm_name(opt.stop_norm)},
		{"mark_scheme", scheme_name(opt.mark_scheme)},
		{"uniform_levels", stop.uniform_levels},
		{"uniform_absolute_tol", stop.uniform_absolute_tol},
		{"uniform_relative_tol", stop.uniform_relative_tol},
		{"uniform_first_err", uni.first_err},
		{"uniform_last_err", uni.last_err},
		{"adaptive_absolute_tol", adaptive_stop.absolute_tol},
		{"adaptive_relative_tol", adaptive_stop.relative_tol},
		{"max_iter", stop.max_iter},
		{"max_dofs", stop.max_dofs},
	}.dump(2) << "\n";
}

// L2 and error marking / L2 stopping have no suffix, so earlier results stay where they are
inline std::string run_dir_name(const StopCriteria &stop, const Options &opt, const int n_threads)
{
	return criteria_dir_name(stop) + "_threads" + std::to_string(n_threads)
		   + (opt.mark_norm == EstimatorNorm::L2 ? "" : "_markh1")
		   + (opt.stop_norm == EstimatorNorm::L2 ? "" : "_stoph1")
		   + (opt.mark_scheme == MarkScheme::ErrorPercentage ? "" : "_element_percentage");
}

inline void print_stop_criteria(const StopCriteria &stop, const Options &opt, const int n_threads,
								const UniformResult &uni, const StopCriteria &adaptive_stop)
{
	std::cout << "stopping criteria (" << mode_name(stop.mode) << ", " << n_threads << " threads, marking "
			  << norm_name(opt.mark_norm) << ", " << scheme_name(opt.mark_scheme) << ", stopping on " << norm_name(opt.stop_norm) << "): uniform levels " << stop.uniform_levels
			  << ", uniform err " << uni.first_err << " -> " << uni.last_err
			  << ", adaptive absolute_tol " << adaptive_stop.absolute_tol
			  << ", relative_tol " << adaptive_stop.relative_tol << ", max dofs " << stop.max_dofs << std::endl;
}
