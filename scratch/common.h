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

using namespace polyfem;
namespace fs = std::filesystem;

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

// <working_dir>/<mesh file name without extension>/
inline fs::path mesh_output_dir(const std::string &working_dir, const std::string &mesh_path)
{
	const fs::path dir = fs::path(working_dir) / fs::path(mesh_path).stem();
	fs::create_directories(dir);
	return dir;
}

// Sends PolyFEM's own outputs (paraview solutions, ...) into <dir>/solutions.
inline json route_outputs(json args, const fs::path &dir, const std::string &tag)
{
	const fs::path sol_dir = dir / "solutions";
	fs::create_directories(sol_dir);
	args["output"]["directory"] = fs::absolute(sol_dir).string();
	args["output"]["paraview"]["file_name"] = tag + ".vtu";
	return args;
}

// geometry can be a single object or an array of objects
inline json &first_geometry(json &args)
{
	return args["geometry"].is_array() ? args["geometry"][0] : args["geometry"];
}

// Works for any triangle / tet mesh (conforming or non-conforming). Must be prepared first.
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

inline std::unique_ptr<mesh::Mesh> load_nc_mesh(State &state, json args)
{
	using Clk = std::chrono::steady_clock;
	const auto secs = [](const Clk::time_point a, const Clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };

	// State::init strips root_path from args, so grab it first
	const std::string root = args.contains("root_path") ? args["root_path"].get<std::string>() : std::string();

	const auto t0 = Clk::now();
	state.init(args, /*strict_validation=*/true);
	const auto t1 = Clk::now();
	// pushes the validated args into the varform; the mesh it loads is replaced by the one below
	state.load_mesh(/*non_conforming=*/true);
	const auto t2 = Clk::now();

	Units units;
	units.init(state.args["units"]);
	const io::FileSystemIO resources(root);
	auto mesh = mesh::GeometryLoader(units, resources)
					.load_fem(state.args["geometry"], /*non_conforming=*/true);
	const auto t3 = Clk::now();

	std::cout << "load_nc_mesh: init " << secs(t0, t1) << " s, load_mesh " << secs(t1, t2)
			  << " s, load_fem " << secs(t2, t3) << " s" << std::endl;
	return mesh;
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

// ======================================================================
// Input args
// ======================================================================

// Franke / Laplacian problem on any mesh file
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

// Reads a PolyFEM args file. Like PolyFEM's own main, it sets root_path to the file when the
// json does not, so relative paths inside it (the mesh, "common", ...) resolve against the file.
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

// ======================================================================
// Marking
// ======================================================================

// Doerfler (bulk) marking: smallest set of elements whose eta^2 sums to >= theta * total.
inline std::vector<int> mark_bulk(const Eigen::VectorXd &eta2, const double theta)
{
	std::vector<int> idx(eta2.size());
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return eta2(a) > eta2(b); });
	const double target = theta * eta2.sum();
	double acc = 0;
	std::vector<int> marked;
	for (int i : idx)
	{
		marked.push_back(i);
		if ((acc += eta2(i)) >= target)
			break;
	}
	return marked;
}