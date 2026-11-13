#pragma once

#include <polyfem/State.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/mesh/GeometryLoader.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <paraviewo/VTUWriter.hpp>

#include <cfenv>

// hypre built with HYPRE_ENABLE_FPE_TRAP (kirby's Debug build) turns on floating-point traps on Linux
// when it initialises. Harmless NaN checks later on (e.g. the units library while materials are
// reloaded after refinement) then kill the program with SIGFPE. Called after every solve.
inline void disable_fp_traps()
{
#if defined(__linux__)
	fedisableexcept(FE_ALL_EXCEPT);
#endif
}

using namespace polyfem;
namespace fs = std::filesystem;

// ======================================================================
// Settings shared by all tests
// ======================================================================

// Non-uniform refinement refines elements below x[axis] = position
struct Plane
{
	int axis = 0;
	double position = 0.5;
	bool use_barycenter = true; // false: any vertex below the plane counts
};

struct Result
{
	int level;
	double h;
	double l2;
	double h1s;
};

enum class Method
{
	Conforming,
	NCUniform,
	NCNonUniform
};

inline std::string method_name(const Method m)
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
// Output helpers
// ======================================================================

inline fs::path mesh_output_dir(const std::string &working_dir, const std::string &mesh_path)
{
	const fs::path dir = fs::path(working_dir) / fs::path(mesh_path).stem();
	fs::create_directories(dir);
	return dir;
}

inline json route_outputs(json args, const fs::path &dir, const std::string &tag)
{
	const fs::path sol_dir = dir / "solutions";
	fs::create_directories(sol_dir);
	args["output"]["directory"] = fs::absolute(sol_dir).string();
	args["output"]["paraview"]["file_name"] = tag + ".vtu";
	return args;
}

inline json &first_geometry(json &args)
{
	return args["geometry"].is_array() ? args["geometry"][0] : args["geometry"];
}

// The mesh must be prepared.
inline void dump_mesh_vtu(const mesh::Mesh &m, const fs::path &path)
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

inline void dump_mesh_vtu(const mesh::Mesh *m, const fs::path &path)
{
	dump_mesh_vtu(*m, path);
}

// ======================================================================
// Non-conforming mesh loading / refinement
// ======================================================================

inline std::unique_ptr<mesh::Mesh> load_fem_mesh(State &state, json args, const bool non_conforming)
{
	using Clk = std::chrono::steady_clock;
	const auto secs = [](const Clk::time_point a, const Clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };

	// State::init strips root_path from args, so grab it first
	const std::string root = args.contains("root_path") ? args["root_path"].get<std::string>() : std::string();

	const auto t0 = Clk::now();
	state.init(args, /*strict_validation=*/true);
	const auto t1 = Clk::now();
	// pushes the validated args into the varform; the mesh it loads is replaced by the one below
	state.load_mesh(non_conforming);
	const auto t2 = Clk::now();

	Units units;
	units.init(state.args["units"]);
	const io::FileSystemIO resources(root);
	auto mesh = mesh::GeometryLoader(units, resources)
					.load_fem(state.args["geometry"], non_conforming);
	const auto t3 = Clk::now();

	std::cout << (non_conforming ? "load_nc_mesh" : "load_conforming_mesh") << ": init " << secs(t0, t1) << " s, load_mesh " << secs(t1, t2)
			  << " s, load_fem " << secs(t2, t3) << " s" << std::endl;
	return mesh;
}

inline std::unique_ptr<mesh::Mesh> load_nc_mesh(State &state, json args)
{
	return load_fem_mesh(state, std::move(args), /*non_conforming=*/true);
}

// Inside f, query the mesh through a mesh::Mesh reference: NCMesh2D/3D hide Mesh::n_elements().
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

// ======================================================================
// Input args
// ======================================================================

inline json make_args(const std::string &mesh_path)
{
	json args = R"json({
        "geometry": [
            {
                "mesh": "",
                "type": "mesh",
                "advanced": { "normalize_mesh": false },
                "n_refs": 0,
                "surface_selection": 1
            }
        ],
        "materials": [ { "type": "Laplacian" } ],
        "boundary_conditions": {
            "rhs": 0,
            "dirichlet_boundary": [
                {
                    "id": "all",
                    "value": "(x^2+y^2)^(1/3)*sin(2/3*(atan2(y,x)+if(y,0,6.283185307179586)))"
                }
            ]
        },
        "output": {
            "log": { "level": 1 },
            "paraview": { "vismesh_rel_area": 0.00001, "file_name": "" },
            "reference": {
                "solution":
                    "(x^2+y^2)^(1/3)*sin(2/3*(atan2(y,x)+if(y,0,6.283185307179586)))"
                ,
                "gradient": [
                    "-2/3*(x^2+y^2)^(-1/6)*sin((atan2(y,x)+if(y,0,6.283185307179586))/3)",
                    "2/3*(x^2+y^2)^(-1/6)*cos((atan2(y,x)+if(y,0,6.283185307179586))/3)"
                ]
            }
        }
    })json"_json;

	json franke_args = R"({
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
	franke_args["geometry"][0]["mesh"] = mesh_path;
	args["space"]["advanced"]["bc_method"] = "lsq";
	franke_args["space"]["advanced"]["bc_method"] = "lsq";
	return franke_args;
}

// root_path defaults to the file, so relative paths inside it resolve against it
inline json load_args(const std::string &json_file)
{
	std::ifstream file(json_file);
	if (!file.is_open())
		log_and_throw_error("Could not open args file {}", json_file);

	json args;
	try
	{
		args = json::parse(file, /*cb=*/nullptr, /*allow_exceptions=*/true, /*ignore_comments=*/true);
	}
	catch (const json::parse_error &err)
	{
		log_and_throw_error("Could not parse {}: {}", json_file, err.what());
	}

	if (!args.contains("root_path"))
		args["root_path"] = fs::absolute(json_file).string();
	return args;
}

// exact_presets: the classes with has_exact_sol() in src/polyfem/problem
inline bool has_reference(const json &args)
{
	static const std::vector<std::string> exact_presets = {
		"Linear", "Quadratic", "Cubic", "Sine", "Franke", "FrankeOld",
		"ElasticExact", "ElasticCantileverExact", "CompressionElasticExact",
		"QuadraticElasticExact", "LinearElasticExact"};
	if (args.contains("preset_problem") && args["preset_problem"].contains("type"))
	{
		const std::string type = args["preset_problem"]["type"];
		if (std::find(exact_presets.begin(), exact_presets.end(), type) != exact_presets.end())
			return true;
	}

	if (!args.contains("output") || !args["output"].contains("reference"))
		return false;
	const json &ref = args["output"]["reference"];
	if (!ref.contains("solution") || ref["solution"].is_null())
		return false;
	const json &sol = ref["solution"];
	if (sol.is_string())
		return !sol.get<std::string>().empty();
	return sol.is_array() && !sol.empty();
}
