#pragma once

#include "common.h"

// ======================================================================
// Angles (degrees). Work on any 2D or 3D simplicial mesh. The mesh must be prepared.
// Triangle interior angles: the single triangle in 2D, all 4 faces of each tet in 3D.
// ======================================================================

inline double min_angle(const mesh::Mesh &mesh)
{
	const int dim = mesh.dimension();
	double result = std::numeric_limits<double>::infinity();

	for (int e = 0; e < mesh.n_elements(); ++e)
	{
		Eigen::Vector3d p[4];
		for (int i = 0; i <= dim; ++i)
		{
			p[i].setZero();
			p[i].head(dim) = mesh.point(mesh.element_vertex(e, i)).transpose();
		}

		const int tris[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
		const int n_tris = (dim == 2) ? 1 : 4;
		for (int t = 0; t < n_tris; ++t)
			for (int c = 0; c < 3; ++c)
			{
				const Eigen::Vector3d u = p[tris[t][(c + 1) % 3]] - p[tris[t][c]];
				const Eigen::Vector3d v = p[tris[t][(c + 2) % 3]] - p[tris[t][c]];
				const double angle = std::atan2(u.cross(v).norm(), u.dot(v)) * 180.0 / M_PI;
				result = std::min(result, angle);
			}
	}
	return result;
}

inline double max_angle(const mesh::Mesh &mesh)
{
	const int dim = mesh.dimension();
	double result = -std::numeric_limits<double>::infinity();

	for (int e = 0; e < mesh.n_elements(); ++e)
	{
		Eigen::Vector3d p[4];
		for (int i = 0; i <= dim; ++i)
		{
			p[i].setZero();
			p[i].head(dim) = mesh.point(mesh.element_vertex(e, i)).transpose();
		}

		const int tris[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
		const int n_tris = (dim == 2) ? 1 : 4;
		for (int t = 0; t < n_tris; ++t)
			for (int c = 0; c < 3; ++c)
			{
				const Eigen::Vector3d u = p[tris[t][(c + 1) % 3]] - p[tris[t][c]];
				const Eigen::Vector3d v = p[tris[t][(c + 2) % 3]] - p[tris[t][c]];
				const double angle = std::atan2(u.cross(v).norm(), u.dot(v)) * 180.0 / M_PI;
				result = std::max(result, angle);
			}
	}
	return result;
}

inline double max_dihedral(const mesh::Mesh &mesh)
{
	double result = -std::numeric_limits<double>::infinity();

	for (int e = 0; e < mesh.n_elements(); ++e)
	{
		Eigen::Vector3d p[4];
		for (int i = 0; i < 4; ++i)
			p[i] = mesh.point(mesh.element_vertex(e, i)).transpose();

		// Face i is the face opposite vertex i
		const int faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
		Eigen::Vector3d n[4];
		for (int f = 0; f < 4; ++f)
		{
			n[f] = (p[faces[f][1]] - p[faces[f][0]]).cross(p[faces[f][2]] - p[faces[f][0]]);
			n[f].normalize();
		}

		// Each pair of faces shares one edge
		for (int i = 0; i < 4; ++i)
			for (int j = i + 1; j < 4; ++j)
			{
				// Dihedral angle = pi - angle between outward normals
				const double ang = M_PI - std::atan2(n[i].cross(n[j]).norm(), n[i].dot(n[j]));
				result = std::max(result, ang * 180.0 / M_PI);
			}
	}
	return result;
}

inline double tri_shape_ratio(const Eigen::Vector3d p[3])
{
	const Eigen::Vector3d e01 = p[1] - p[0];
	const Eigen::Vector3d e02 = p[2] - p[0];

	const double area = 0.5 * e01.cross(e02).norm();
	if (area <= 0)
		return std::numeric_limits<double>::infinity();

	const double a = (p[1] - p[2]).norm();
	const double b = (p[0] - p[2]).norm();
	const double c = (p[0] - p[1]).norm();
	const double s = 0.5 * (a + b + c);

	const double rho = area / s; // incircle radius
	if (rho <= 0)
		return std::numeric_limits<double>::infinity();

	const double h = std::max({a, b, c}); // triangle diameter = longest side
	return h / (2.0 * rho);
}

inline double tet_shape_ratio(const Eigen::Vector3d p[4])
{
	const Eigen::Vector3d e01 = p[1] - p[0];
	const Eigen::Vector3d e02 = p[2] - p[0];
	const Eigen::Vector3d e03 = p[3] - p[0];

	const double volume = std::abs(e01.dot(e02.cross(e03))) / 6.0;
	if (volume <= 0)
		return std::numeric_limits<double>::infinity();

	const int tris[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
	double total_area = 0;
	for (int t = 0; t < 4; ++t)
	{
		const Eigen::Vector3d u = p[tris[t][1]] - p[tris[t][0]];
		const Eigen::Vector3d v = p[tris[t][2]] - p[tris[t][0]];
		total_area += 0.5 * u.cross(v).norm();
	}

	const double rho = 3.0 * volume / total_area; // inscribed sphere radius
	if (rho <= 0)
		return std::numeric_limits<double>::infinity();

	double h = 0; // tet diameter = longest edge
	for (int i = 0; i < 4; ++i)
		for (int j = i + 1; j < 4; ++j)
			h = std::max(h, (p[i] - p[j]).norm());

	return h / (2.0 * rho); // h_T / rho_T, rho_T = diameter of inscribed sphere
}

// Max sigma_T over all elements, triangle (2D) or tet (3D).
inline double max_shape_ratio(const mesh::Mesh &mesh)
{
	const int dim = mesh.dimension();
	double result = 0;

	for (int e = 0; e < mesh.n_elements(); ++e)
	{
		Eigen::Vector3d p[4];
		for (int i = 0; i <= dim; ++i)
		{
			p[i].setZero();
			p[i].head(dim) = mesh.point(mesh.element_vertex(e, i)).transpose();
		}

		const double sr = (dim == 2) ? tri_shape_ratio(p) : tet_shape_ratio(p);
		result = std::max(result, sr);
	}
	return result;
}

// ======================================================================
// Refinement sequence
// ======================================================================

// Loads the NC mesh, applies opt.base_refs, then calls on_iteration(mesh, 0).
// For each iteration 1..iterations it refines once (uniformly, or only below the plane)
// and calls on_iteration(mesh, i). The mesh is always prepared when the callback runs.
inline void refinement_sequence(const fs::path &dir, const std::string &tag, const json &args,
								const bool nonuniform, const int iterations, const Options &opt,
								const std::function<void(const mesh::Mesh &, int)> &on_iteration)
{
	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, route_outputs(args, dir, tag));

	visit_nc_mesh(*m, [&](auto &nc) {
		const mesh::Mesh &base = nc;

		nc.refine(opt.base_refs, 0.5);
		nc.prepare_mesh();
		on_iteration(base, 0);

		for (int i = 1; i <= iterations; ++i)
		{
			if (nonuniform)
			{
				refine_below_plane(nc, opt.plane);
				nc.refine(1, 0.5);
				nc.prepare_mesh();
			}
			else
			{
				nc.refine(1, 0.5);
				nc.prepare_mesh();
			}
			on_iteration(base, i);
		}
	});
}

// ======================================================================
// Test: angles + element counts per refinement iteration  ->  angles_<uniform|nonuniform>.csv
// ======================================================================

inline void angle_test(const fs::path &dir, const json &args, const bool nonuniform,
					   const int iterations, const Options &opt)
{
	const std::string tag = std::string("angles_") + (nonuniform ? "nonuniform" : "uniform");
	fs::create_directories(dir);

	std::ofstream csv(dir / (tag + ".csv"));
	csv.precision(17);
	csv << "iteration,n_elements,min_angle,max_angle,max_shape_ratio\n";

	refinement_sequence(dir, tag, args, nonuniform, iterations, opt,
						[&](const mesh::Mesh &m, const int it) {
							const double lo = min_angle(m);
							double hi = max_angle(m);
							hi = max_dihedral(m);
							const double sr = max_shape_ratio(m);
							csv << it << "," << m.n_elements() << "," << lo << "," << hi << "," << sr << "\n";
							std::cout << tag << " iter " << it << ": elements " << m.n_elements()
									  << ", min angle " << lo << ", max angle " << hi
									  << ", max shape ratio " << sr << std::endl;
							if (opt.dump_meshes)
								dump_mesh_vtu(m, dir / "meshes" / (tag + "_iter" + std::to_string(it) + ".vtu"));
						});
}

// ======================================================================
// Test: only the mesh dumps for each refinement iteration (no angles)
// ======================================================================

inline void dump_refinement_meshes(const fs::path &dir, const json &args, const bool nonuniform,
								   const int iterations, const Options &opt)
{
	const std::string tag = std::string("refinement_") + (nonuniform ? "nonuniform" : "uniform");
	refinement_sequence(dir, tag, args, nonuniform, iterations, opt,
						[&](const mesh::Mesh &m, const int it) {
							dump_mesh_vtu(m, dir / "meshes" / (tag + "_iter" + std::to_string(it) + ".vtu"));
						});
}

// ======================================================================
// Test: convergence (L2 / H1-semi errors vs h) for conforming, NC uniform, NC non-uniform
// ======================================================================

inline Result solve_conforming(const fs::path &dir, json args, const int p, const int level, const Options &opt)
{
	const std::string tag = "conforming_p" + std::to_string(p) + "_L" + std::to_string(level);
	args["space"]["discr_order"] = p;
	first_geometry(args)["n_refs"] = opt.base_refs + level;
	args = route_outputs(args, dir, tag);

	State state;
	state.init(args, /*strict_validation=*/true);
	state.load_mesh(/*non_conforming=*/false);

	Eigen::MatrixXd sol;
	state.solve(sol);
	std::cout << tag << ": max |sol| = " << sol.cwiseAbs().maxCoeff()
			  << ", min = " << sol.minCoeff() << ", max = " << sol.maxCoeff() << std::endl;

	if (opt.dump_meshes)
		dump_mesh_vtu(state.variational_formulation->output_space().mesh, dir / "meshes" / (tag + ".vtu"));

	state.variational_formulation->export_data(sol);

	const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
	return {level, stats.mesh_size, stats.l2_err, stats.h1_semi_err};
}

// base_refs -> (non-uniform passes) -> `level` uniform refinements -> solve
inline Result solve_nc(const fs::path &dir, json args, const int p, const int level,
					   const bool nonuniform, const Options &opt)
{
	const std::string tag = method_name(nonuniform ? Method::NCNonUniform : Method::NCUniform)
							+ "_p" + std::to_string(p) + "_L" + std::to_string(level);
	args["space"]["discr_order"] = p;
	args = route_outputs(args, dir, tag);

	State state;
	std::unique_ptr<mesh::Mesh> m = load_nc_mesh(state, args);

	visit_nc_mesh(*m, [&](auto &nc) {
		nc.refine(opt.base_refs, 0.5);
		if (nonuniform)
			for (int i = 0; i < opt.nonuniform_passes; ++i)
				refine_below_plane(nc, opt.plane);
		nc.refine(level, 0.5);
		nc.prepare_mesh();
	});

	if (opt.dump_meshes)
		dump_mesh_vtu(*m, dir / "meshes" / (tag + ".vtu"));

	state.variational_formulation->set_mesh(std::move(m));

	Eigen::MatrixXd sol;
	state.solve(sol);
	std::cout << tag << ": max |sol| = " << sol.cwiseAbs().maxCoeff()
			  << ", min = " << sol.minCoeff() << ", max = " << sol.maxCoeff() << std::endl;
	state.variational_formulation->export_data(sol);

	const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);
	return {level, stats.mesh_size, stats.l2_err, stats.h1_semi_err};
}

inline double rate(const double e_prev, const double e, const double h_prev, const double h)
{
	return std::log(e_prev / e) / std::log(h_prev / h);
}

// Least-squares slope of log(err) vs log(h) over all points (the convergence order).
// Returns NaN if fewer than 2 usable points.
inline double fit_slope(const std::vector<Result> &rs, const bool use_l2)
{
	std::vector<double> x, y;
	for (const Result &r : rs)
	{
		const double err = use_l2 ? r.l2 : r.h1s;
		if (r.h > 0 && err > 0)
		{
			x.push_back(std::log(r.h));
			y.push_back(std::log(err));
		}
	}
	if (x.size() < 2)
		return std::numeric_limits<double>::quiet_NaN();

	double mx = 0, my = 0;
	for (size_t i = 0; i < x.size(); ++i)
	{
		mx += x[i];
		my += y[i];
	}
	mx /= x.size();
	my /= y.size();

	double sxy = 0, sxx = 0;
	for (size_t i = 0; i < x.size(); ++i)
	{
		sxy += (x[i] - mx) * (y[i] - my);
		sxx += (x[i] - mx) * (x[i] - mx);
	}
	return sxy / sxx;
}

inline void print_table(const std::string &label, const int p, const std::vector<Result> &rs, std::ostream &out)
{
	out << "\n== " << label << "  (expect L2 rate ~" << p + 1 << ", H1-semi rate ~" << p << ")\n";
	out << std::setw(5) << "lvl" << std::setw(13) << "h"
		<< std::setw(13) << "L2" << std::setw(8) << "rate"
		<< std::setw(13) << "H1semi" << std::setw(8) << "rate" << "\n";

	for (size_t i = 0; i < rs.size(); ++i)
	{
		out << std::setw(5) << rs[i].level
			<< std::setw(13) << std::scientific << std::setprecision(3) << rs[i].h
			<< std::setw(13) << rs[i].l2;
		if (i == 0)
			out << std::setw(8) << "-";
		else
			out << std::setw(8) << std::fixed << std::setprecision(2)
				<< rate(rs[i - 1].l2, rs[i].l2, rs[i - 1].h, rs[i].h);

		out << std::setw(13) << std::scientific << std::setprecision(3) << rs[i].h1s;
		if (i == 0)
			out << std::setw(8) << "-";
		else
			out << std::setw(8) << std::fixed << std::setprecision(2)
				<< rate(rs[i - 1].h1s, rs[i].h1s, rs[i - 1].h, rs[i].h);
		out << "\n";
	}

	out << "log-log least-squares slope:  L2 = " << std::fixed << std::setprecision(3) << fit_slope(rs, true)
		<< " (expect " << p + 1 << "),  H1-semi = " << fit_slope(rs, false)
		<< " (expect " << p << ")\n";
}

inline void write_convergence_csv(const fs::path &path, const std::vector<Result> &rs)
{
	std::ofstream csv(path);
	csv.precision(17);
	csv << "level,h,l2,l2_rate,h1_semi,h1_semi_rate\n";
	const double nan = std::numeric_limits<double>::quiet_NaN();
	for (size_t i = 0; i < rs.size(); ++i)
	{
		const double r_l2 = i == 0 ? nan : rate(rs[i - 1].l2, rs[i].l2, rs[i - 1].h, rs[i].h);
		const double r_h1 = i == 0 ? nan : rate(rs[i - 1].h1s, rs[i].h1s, rs[i - 1].h, rs[i].h);
		csv << rs[i].level << "," << rs[i].h << "," << rs[i].l2 << "," << r_l2 << ","
			<< rs[i].h1s << "," << r_h1 << "\n";
	}
}

// Runs levels 0..max_level and writes convergence_<method>_p<p>.txt / .csv
inline std::vector<Result> convergence_test(const fs::path &dir, const json &args, const Method method,
											const int p, const int max_level, const Options &opt)
{
	fs::create_directories(dir);

	std::vector<Result> rs;
	for (int level = 0; level <= max_level; ++level)
	{
		switch (method)
		{
		case Method::Conforming:
			rs.push_back(solve_conforming(dir, args, p, level, opt));
			break;
		case Method::NCUniform:
			rs.push_back(solve_nc(dir, args, p, level, /*nonuniform=*/false, opt));
			break;
		case Method::NCNonUniform:
			rs.push_back(solve_nc(dir, args, p, level, /*nonuniform=*/true, opt));
			break;
		}
	}

	const std::string name = method_name(method) + "_p" + std::to_string(p);
	print_table(name, p, rs, std::cout);
	std::ofstream txt(dir / ("convergence_" + name + ".txt"));
	print_table(name, p, rs, txt);
	write_convergence_csv(dir / ("convergence_" + name + ".csv"), rs);
	return rs;
}

// ======================================================================
// Plotting
// ======================================================================

// Runs plot_results.py on the mesh folder; it writes the PNGs into that same folder.
inline void make_plots(const std::string &plot_script, const fs::path &dir)
{
	if (!fs::exists(plot_script))
	{
		std::cout << "plot script not found, skipping plots: " << plot_script << std::endl;
		return;
	}
	const std::string cmd = "python3 \"" + plot_script + "\" \"" + dir.string() + "\"";
	if (std::system(cmd.c_str()) != 0)
		std::cout << "plot script failed: " << cmd << std::endl;
}