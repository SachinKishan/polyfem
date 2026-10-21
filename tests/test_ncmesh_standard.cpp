////////////////////////////////////////////////////////////////////////////////
// Smoke test: for every JSON listed in tests/standard.txt, load the mesh via
// State::load_mesh(/*non_conforming=*/true) and run solve() to completion.
// No comparison against recorded errors -- just "does it load and solve
// without crashing or throwing" on the non-conforming mesh data structure.
//
// Several formulations are known to SIGSEGV (not throw) when forced through
// NCMesh2D/3D -- a native crash like that takes the whole process down and
// bypasses try/catch entirely, so a single bad file would otherwise stop the
// sweep before later files even run. To keep the sweep going and still
// report every file, each file is run in its own forked child process: the
// parent just waits for it and records PASS / SKIP / FAIL / CRASHED(signal)
// without the child's crash touching the parent at all. On platforms
// without fork() (e.g. native Windows), the test falls back to running
// in-process and stops at the first crash, same as before.
////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>

#include "polyfem/utils/JSONUtils.hpp"
#include "polyfem/State.hpp"
#include "polyfem/varforms/VarForm.hpp"
#include "polyfem/varforms/VarFormFactory.hpp"
#include "polyfem/io/ResourceIO.hpp"
#include "spdlog/spdlog.h"
#include <polyfem/Common.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#define NCMESH_TEST_HAS_FORK 1
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <cstring>
#endif
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::utils;

namespace
{
	bool load_json(const std::string &json_file, json &out)
	{
		std::ifstream file(json_file);
		if (!file.is_open())
			return false;
		file >> out;
		return true;
	}

	enum class Outcome
	{
		Pass,
		Skip,
		Fail,
		Crashed // process died from a signal (SIGSEGV, SIGBUS, ...)
	};

	struct Result
	{
		std::string file;
		Outcome outcome;
		std::string reason; // populated for Skip/Fail/Crashed
	};

	// Runs state.init / load_mesh(non_conforming=true) / solve for one file.
	// Exceptions are caught here; a hard crash (SIGSEGV/SIGBUS) is NOT caught
	// by this function -- that's handled by the fork() wrapper below.
	Result run_one(const std::string &data_dir, const std::string &relative_path)
	{
		const std::string json_file = data_dir + "/" + relative_path;

		json in_args;
		if (!load_json(json_file, in_args))
			return {relative_path, Outcome::Skip, "file not found"};

		json args = in_args;
		args["root_path"] = json_file;
		utils::apply_common_params(args);
		args["/output/log/level"_json_pointer] = "error";

		if (!varform::uses_varform_state(args, io::FileSystemIO(json_file)))
			return {relative_path, Outcome::Skip, "formulation not on the VarForm/State path (legacy-only, e.g. Navier-Stokes/FSI)"};

		State state;
		try
		{
			state.init(args, /*strict_validation=*/true);
			state.set_max_threads(1);
		}
		catch (const std::exception &e)
		{
			return {relative_path, Outcome::Skip, std::string("state.init threw: ") + e.what()};
		}

		try
		{
			state.load_mesh(/*non_conforming=*/true);
		}
		catch (const std::exception &e)
		{
			return {relative_path, Outcome::Skip, std::string("load_mesh(non_conforming=true) threw: ") + e.what()};
		}

		if (state.variational_formulation == nullptr)
			return {relative_path, Outcome::Skip, "no variational_formulation after load_mesh"};

		Eigen::MatrixXd sol;
		try
		{
			state.solve(sol);
		}
		catch (const std::exception &e)
		{
			return {relative_path, Outcome::Fail, std::string("solve() threw: ") + e.what()};
		}

		if (sol.size() <= 0)
			return {relative_path, Outcome::Fail, "empty solution after solve()"};

		if (!sol.allFinite())
			return {relative_path, Outcome::Fail, "solution contains NaN/Inf"};

		return {relative_path, Outcome::Pass, ""};
	}

#ifdef NCMESH_TEST_HAS_FORK
	// Encodes a Result across the fork() boundary as a single line of text on
	// a pipe: "<outcome_code>\t<reason>". The child's own stdout/stderr (all
	// of polyfem's logging) still goes to the parent's terminal normally --
	// only this one summary line travels through the pipe.
	Result run_one_isolated(const std::string &data_dir, const std::string &relative_path)
	{
		int pipefd[2];
		if (pipe(pipefd) != 0)
			return {relative_path, Outcome::Fail, "pipe() failed, running in-process instead"};

		const pid_t pid = fork();
		if (pid < 0)
		{
			close(pipefd[0]);
			close(pipefd[1]);
			return run_one(data_dir, relative_path); // fork failed, fall back
		}

		if (pid == 0)
		{
			// Child: do the actual work, report result through the pipe, then
			// exit directly (not via normal unwinding) so Catch2's own
			// bookkeeping doesn't get duplicated by both processes.
			close(pipefd[0]);
			Result r = run_one(data_dir, relative_path);
			std::ostringstream line;
			line << static_cast<int>(r.outcome) << "\t" << r.reason;
			const std::string s = line.str();
			ssize_t written = write(pipefd[1], s.data(), s.size());
			(void)written;
			close(pipefd[1]);
			std::_Exit(0);
		}

		// Parent: read whatever the child managed to write (empty if it
		// crashed before writing anything), then wait for its exit status.
		close(pipefd[1]);
		std::string buf;
		char chunk[4096];
		ssize_t n;
		while ((n = read(pipefd[0], chunk, sizeof(chunk))) > 0)
			buf.append(chunk, static_cast<size_t>(n));
		close(pipefd[0]);

		int status = 0;
		waitpid(pid, &status, 0);

		if (WIFSIGNALED(status))
		{
			const int sig = WTERMSIG(status);
			return {relative_path, Outcome::Crashed, std::string("terminated by signal ") + std::to_string(sig) + " (" + strsignal(sig) + ")"};
		}

		if (buf.empty())
			return {relative_path, Outcome::Crashed, "child exited without reporting a result (status " + std::to_string(status) + ")"};

		const auto tab = buf.find('\t');
		const Outcome outcome = static_cast<Outcome>(std::stoi(buf.substr(0, tab)));
		const std::string reason = tab == std::string::npos ? "" : buf.substr(tab + 1);
		return {relative_path, outcome, reason};
	}
#endif

	void print_summary(const std::vector<Result> &results)
	{
		int n_pass = 0, n_skip = 0, n_fail = 0, n_crash = 0;
		std::ostringstream out;
		for (const Result &r : results)
		{
			switch (r.outcome)
			{
			case Outcome::Pass:
				++n_pass;
				out << "[PASS   ] " << r.file << "\n";
				break;
			case Outcome::Skip:
				++n_skip;
				out << "[SKIP   ] " << r.file << " (" << r.reason << ")\n";
				break;
			case Outcome::Fail:
				++n_fail;
				out << "[FAIL   ] " << r.file << " (" << r.reason << ")\n";
				break;
			case Outcome::Crashed:
				++n_crash;
				out << "[CRASHED] " << r.file << " (" << r.reason << ")\n";
				break;
			}
		}
		out << "\n" << n_pass << " passed, " << n_skip << " skipped, " << n_fail << " failed, "
			<< n_crash << " crashed (" << results.size() << " total)\n";

		spdlog::info("\n{}", out.str());
		std::cout << out.str();
	}
} // namespace

TEST_CASE("ncmesh standard data smoke test", "[ncmesh][standard]")
{
	spdlog::set_level(spdlog::level::err);

	std::ifstream manifest(std::string(POLYFEM_TEST_DIR) + "/standard.txt");
	REQUIRE(manifest.is_open());

	std::vector<Result> results;
	std::string line;
	while (std::getline(manifest, line))
	{
		if (line.empty() || line[0] == '#' || line[0] == '*')
			continue;
		CAPTURE(line);
		std::cout << "[RUN ] " << line << std::flush << std::endl;
#ifdef NCMESH_TEST_HAS_FORK
		results.push_back(run_one_isolated(POLYFEM_DATA_DIR, line));
#else
		results.push_back(run_one(POLYFEM_DATA_DIR, line));
#endif
		std::cout << "[DONE] " << line << " -> "
				  << (results.back().outcome == Outcome::Pass      ? "PASS"
						  : results.back().outcome == Outcome::Skip    ? "SKIP"
						  : results.back().outcome == Outcome::Fail    ? "FAIL"
																		: "CRASHED")
				  << std::flush << std::endl;
	}

	print_summary(results);

	// Only Fail is a hard test failure here. Crashed files are reported (and
	// stand out loudly in the summary above) but don't fail the CHECK below,
	// since we explicitly want the suite to finish and report on everything
	// -- the point of this run is to see the whole picture, then go look at
	// crashes/failures one at a time.
	std::vector<std::string> failing;
	for (const Result &r : results)
		if (r.outcome == Outcome::Fail)
			failing.push_back(r.file + ": " + r.reason);

	INFO("Failing files:\n" << [&] {
		std::ostringstream s;
		for (const auto &f : failing)
			s << "  " << f << "\n";
		return s.str();
	}());
	CHECK(failing.empty());
}