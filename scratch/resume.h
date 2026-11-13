#pragma once

#include "common.h"

// completed.json: a run is added only after its csv and solution are written, so a stopped run is redone

inline json read_completed(const fs::path &run_dir)
{
	const fs::path f = run_dir / "completed.json";
	if (!fs::exists(f))
		return json::object();
	return json::parse(std::ifstream(f));
}

inline void mark_completed(const fs::path &run_dir, const std::string &tag, const double first_err, const double last_err,
						   const std::string &stop_reason)
{
	json done = read_completed(run_dir);
	done[tag] = {{"first_err", first_err}, {"last_err", last_err}, {"stop_reason", stop_reason}};
	std::ofstream(run_dir / "completed.json") << done.dump(2) << "\n";
}

// nan is written as null
inline double json_err(const json &j)
{
	return j.is_number() ? j.get<double>() : std::numeric_limits<double>::quiet_NaN();
}
