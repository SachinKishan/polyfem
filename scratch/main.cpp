#include <polyfem/State.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <paraviewo/VTUWriter.hpp>

using namespace polyfem;
namespace fs = std::filesystem;

// Not in an anonymous namespace on purpose, so functions you don't call from main
// don't produce unused-function warnings.

	// ======================================================================
	// Settings shared by all tests
	// ======================================================================

	// Region used by the non-uniform refinement: elements on the negative side of the
	// plane  x[axis] = position  get refined.
	//   use_barycenter = true  -> an element is refined if its barycenter is below the plane
	//   use_barycenter = false -> an element is refined if ANY of its vertices is below the plane
	struct Plane
	{
		int axis = 0;
		double position = 0.5;
		bool use_barycenter = true;
	};

	struct Options
	{
		int base_refs = 1;          // uniform splits applied to the input mesh before anything else
		int nonuniform_passes = 1;  // convergence tests only: non-uniform passes applied before the uniform levels
		Plane plane;                // region for non-uniform refinement
		bool dump_meshes = true;    // write .vtu mesh dumps
	};

	struct Result
	{
		int level;
		double h;   // max edge length (OutStatsData::mesh_size)
		double l2;  // expected O(h^{p+1})
		double h1s; // expected O(h^p)
	};

	enum class Method
	{
		Conforming,
		NCUniform,
		NCNonUniform
	};

	std::string method_name(const Method m)
	{
		switch (m)
		{
		case Method::Conforming:
			return "conforming";
		case Method::NCUniform:
			return "nc_uniform";
		case Method::NCNonUniform:
			return "nc_nonuniform";
		}
		return "";
	}

	// ======================================================================
	// Angles (degrees). Work on any 2D or 3D simplicial mesh. The mesh must be prepared.
	// Triangle interior angles: the single triangle in 2D, all 4 faces of each tet in 3D.
	// ======================================================================

	double min_angle(const mesh::Mesh &mesh)
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

	double max_angle(const mesh::Mesh &mesh)
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

double tri_shape_ratio(const Eigen::Vector3d p[3])
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
double tet_shape_ratio(const Eigen::Vector3d p[4])
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
double max_shape_ratio(const mesh::Mesh &mesh)
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
	// Output helpers
	// ======================================================================

	// <working_dir>/<mesh file name without extension>/
	fs::path mesh_output_dir(const std::string &working_dir, const std::string &mesh_path)
	{
		const fs::path dir = fs::path(working_dir) / fs::path(mesh_path).stem();
		fs::create_directories(dir);
		return dir;
	}

	// Sends PolyFEM's own outputs (paraview solutions, ...) into <dir>/solutions.
	json route_outputs(json args, const fs::path &dir, const std::string &tag)
	{
		const fs::path sol_dir = dir / "solutions";
		fs::create_directories(sol_dir);
		args["output"]["directory"] = fs::absolute(sol_dir).string();
		args["output"]["paraview"]["file_name"] = tag + ".vtu";
		return args;
	}

	// geometry can be a single object or an array of objects
	json &first_geometry(json &args)
	{
		return args["geometry"].is_array() ? args["geometry"][0] : args["geometry"];
	}

	// Works for any triangle / tet mesh (conforming or non-conforming). Must be prepared first.
	void dump_mesh_vtu(const mesh::Mesh &m, const fs::path &path)
	{
		fs::create_directories(path.parent_path());

		const int dim = m.dimension();
		if (m.n_elements() > 0 && m.n_cell_vertices(0) != dim + 1)
			log_and_throw_error("dump_mesh_vtu only supports triangle / tet meshes.");

		Eigen::MatrixXd V(m.n_vertices(), dim);
		for (int i = 0; i < m.n_vertices(); ++i)
			V.row(i) = m.point(i);

		Eigen::MatrixXi F(m.n_elements(), dim + 1);
		for (int e = 0; e < m.n_elements(); ++e)
			for (int lv = 0; lv <= dim; ++lv)
				F(e, lv) = m.element_vertex(e, lv);

		paraviewo::VTUWriter writer;
		writer.write_mesh(path.string(), V, F,
						  dim == 2 ? paraviewo::CellType::Triangle : paraviewo::CellType::Tetrahedron);
		std::cout << "wrote mesh dump " << path << std::endl;
	}

	void dump_mesh_vtu(const mesh::Mesh *m, const fs::path &path)
	{
		dump_mesh_vtu(*m, path);
	}

	// ======================================================================
	// Non-conforming mesh loading / refinement
	// ======================================================================

	std::unique_ptr<mesh::Mesh> load_nc_mesh(State &state, json args)
	{
		state.init(args, /*strict_validation=*/true);
		// pushes the validated args into the varform; the mesh it loads is replaced by the one below
		state.load_mesh(/*non_conforming=*/true);

		Units units;
		units.init(state.args["units"]);
		return mesh::read_fem_geometry(
			units, state.args["geometry"], state.args["root_path"],
			{}, {}, {}, /*non_conforming=*/true);
	}

	// Calls f(nc) with the concrete NCMesh2D or NCMesh3D. Inside f, query the mesh through a
	// mesh::Mesh reference: NCMesh2D/3D have a protected member called n_elements that hides
	// Mesh::n_elements().
	template <class F>
	void visit_nc_mesh(mesh::Mesh &m, F &&f)
	{
		if (auto *nc3 = dynamic_cast<mesh::NCMesh3D *>(&m))
			f(*nc3);
		else if (auto *nc2 = dynamic_cast<mesh::NCMesh2D *>(&m))
			f(*nc2);
		else
			log_and_throw_error("Expected an NCMesh2D or NCMesh3D.");
	}

	// Refines every element on the negative side of the plane, then rebuilds the index maps.
	template <class NC>
	void refine_below_plane(NC &nc, const Plane &plane)
	{
		mesh::Mesh &base = nc;
		const int dim = base.dimension();
		if (plane.axis < 0 || plane.axis >= dim)
			log_and_throw_error("Plane axis is out of range for this mesh dimension.");

		nc.prepare_mesh(); // refine_elements needs up-to-date valid-element index maps

		std::vector<int> ids;
		for (int e = 0; e < base.n_elements(); ++e)
		{
			double lowest = std::numeric_limits<double>::infinity();
			double mean = 0;
			for (int lv = 0; lv <= dim; ++lv)
			{
				const double x = base.point(base.element_vertex(e, lv))(plane.axis);
				lowest = std::min(lowest, x);
				mean += x;
			}
			mean /= (dim + 1);

			if ((plane.use_barycenter ? mean : lowest) < plane.position)
				ids.push_back(e);
		}

		nc.refine_elements(ids);
		nc.prepare_mesh();
	}

	// Loads the NC mesh, applies opt.base_refs, then calls on_iteration(mesh, 0).
	// For each iteration 1..iterations it refines once (uniformly, or only below the plane)
	// and calls on_iteration(mesh, i). The mesh is always prepared when the callback runs.
	void refinement_sequence(const fs::path &dir, const std::string &tag, const json &args,
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
					nc.refine(1,0.5);
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
	// Test 1: angles + element counts per refinement iteration  ->  angles_<uniform|nonuniform>.csv
	// ======================================================================

	void angle_test(const fs::path &dir, const json &args, const bool nonuniform,
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
								const double hi = max_angle(m);
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
	// Test 2: only the mesh dumps for each refinement iteration (no angles)
	// ======================================================================

	void dump_refinement_meshes(const fs::path &dir, const json &args, const bool nonuniform,
								const int iterations, const Options &opt)
	{
		const std::string tag = std::string("refinement_") + (nonuniform ? "nonuniform" : "uniform");
		refinement_sequence(dir, tag, args, nonuniform, iterations, opt,
							[&](const mesh::Mesh &m, const int it) {
								dump_mesh_vtu(m, dir / "meshes" / (tag + "_iter" + std::to_string(it) + ".vtu"));
							});
	}

	// ======================================================================
	// Test 3: convergence (L2 / H1-semi errors vs h) for conforming, NC uniform, NC non-uniform
	// ======================================================================

	Result solve_conforming(const fs::path &dir, json args, const int p, const int level, const Options &opt)
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
	Result solve_nc(const fs::path &dir, json args, const int p, const int level,
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

	double rate(const double e_prev, const double e, const double h_prev, const double h)
	{
		return std::log(e_prev / e) / std::log(h_prev / h);
	}

	// Least-squares slope of log(err) vs log(h) over all points (the convergence order).
	// Returns NaN if fewer than 2 usable points.
	double fit_slope(const std::vector<Result> &rs, const bool use_l2)
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

	void print_table(const std::string &label, const int p, const std::vector<Result> &rs, std::ostream &out)
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

	void write_convergence_csv(const fs::path &path, const std::vector<Result> &rs)
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
	std::vector<Result> convergence_test(const fs::path &dir, const json &args, const Method method,
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
	// Input args and plotting
	// ======================================================================

	// Franke / Laplacian problem on any mesh file
	json make_args(const std::string &mesh_path)
	{
		json args = R"({
			"geometry": [
				{
					"mesh": "",
					"type": "mesh",
					"advanced": { "normalize_mesh": true },
					"n_refs": 0,
					"surface_selection": 1
				}
			],
			"materials": [ { "type": "Laplacian" } ],
			"preset_problem": { "type": "Franke" },
			"output": {
				"log": { "level": 1 },
				"paraview": { "vismesh_rel_area": 0.00001, "file_name": "" }
			}
		})"_json;

		args["geometry"][0]["mesh"] = mesh_path;
		args["space"]["advanced"]["bc_method"] = "lsq";
		return args;
	}

	// Runs plot_results.py on the mesh folder; it writes the PNGs into that same folder.
	void make_plots(const std::string &plot_script, const fs::path &dir)
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


int main()
{
	// ====================== edit freely ======================
	const std::string working_dir = "/Users/sak9889/WORK/Projects/BETTER_ITR/WorkingDirectory";
	const std::string plot_script = working_dir + "/plot_results.py"; // copy plot_results.py here

	Options opt;
	opt.base_refs = 0;
	opt.nonuniform_passes = 1;
	opt.plane = {/*axis=*/0, /*position=*/0.5, /*use_barycenter=*/true};
	opt.dump_meshes = true;
	// =========================================================

	fs::create_directories(working_dir);

	// ---- per mesh: pick the file, get its folder + args, then call whatever you want ----
	const std::string mesh = "/Users/sak9889/WORK/Projects/BETTER_ITR/meshes/simple_tet.msh";
	//const std::string mesh = "//Users/sak9889/WORK/Projects/BETTER_ITR/meshes/tet_with_v6v9_long.msh";

		const json args = make_args(mesh);
	const fs::path dir = mesh_output_dir(working_dir, mesh); // <working_dir>/simple_tet/
	std::ofstream(dir / "args.json") << args.dump(2) << "\n";

	// angles + element counts per refinement iteration (also dumps a .vtu per iteration)
	//angle_test(dir, args, /*nonuniform=*/false, /*iterations=*/3, opt);
	//angle_test(dir, args, /*nonuniform=*/true, /*iterations=*/3, opt);

	// only mesh dumps per iteration
	// dump_refinement_meshes(dir, args, /*nonuniform=*/false, 5, opt);
	// dump_refinement_meshes(dir, args, /*nonuniform=*/true, 5, opt);

	// convergence (p, max_level)
	convergence_test(dir, args, Method::Conforming, 2, 3, opt);
	convergence_test(dir, args, Method::NCUniform, 1, 3, opt);
	convergence_test(dir, args, Method::NCUniform, 2, 3, opt);
	//convergence_test(dir, args, Method::NCNonUniform, 1, 3, opt);

	// single solves, if you want to build your own sweeps
	// Result r = solve_nc(dir, args, /*p=*/1, /*level=*/2, /*nonuniform=*/true, opt);

	// plots for everything currently in the folder (angles_*.csv, convergence_*.csv)
	make_plots(plot_script, dir);

	// ---- another mesh: same pattern ----
	// const std::string mesh2 = "/Users/sak9889/WORK/GitHub/Polyfem/polyfem-data/contact/meshes/2D/simple/square.obj";
	// const json args2 = make_args(mesh2);
	// const fs::path dir2 = mesh_output_dir(working_dir, mesh2);
	// angle_test(dir2, args2, false, 5, opt);
	// make_plots(plot_script, dir2);

	return EXIT_SUCCESS;
}